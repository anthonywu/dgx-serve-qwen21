"""Persistent, single-worker Qwen Image 2.1 HTTP server."""
import asyncio
import base64
from contextlib import asynccontextmanager
from io import BytesIO
import logging
from pathlib import Path
import secrets
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen-Image-2.1'
log = logging.getLogger('uvicorn.error')
pipeline = None
lock = asyncio.Lock()
loaded_at = None


def load_model():
    import torch
    from diffusers import QwenImage21Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing CPU fallback.')
    log.info('Loading local BF16 weights onto %s', torch.cuda.get_device_name())
    model = QwenImage21Pipeline.from_pretrained(
        str(ROOT / 'model'), torch_dtype=torch.bfloat16, local_files_only=True,
    ).to('cuda')
    model.vae.enable_tiling()
    log.info('Weights resident: %.2f GiB CUDA allocated, %.2f GiB reserved',
             torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3)
    return model


@asynccontextmanager
async def lifespan(app):
    global pipeline, loaded_at
    pipeline = await asyncio.to_thread(load_model)
    loaded_at = time.time()
    log.info('Model loaded. Ready for requests; weights remain resident.')
    yield
    pipeline = None


app = FastAPI(title='Qwen Image 2.1', lifespan=lifespan)
app.mount('/samples', StaticFiles(directory=ROOT / 'samples'), name='samples')


@app.get('/', include_in_schema=False)
async def studio():
    return FileResponse(ROOT / 'index.html')


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    width: int = Field(default=1024, ge=512, le=2752)
    height: int = Field(default=1024, ge=512, le=2752)
    steps: int = Field(default=40, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @model_validator(mode='after')
    def dimensions(self):
        if self.width % 32 or self.height % 32:
            raise ValueError('Width and height must be multiples of 32.')
        if self.width * self.height > 4_400_000:
            raise ValueError('Maximum image area is 4.4 megapixels.')
        return self


class OpenAIRequest(GenerateRequest):
    model: str = MODEL_ID
    n: int = Field(default=1, ge=1, le=1)
    size: str | None = None
    response_format: str = 'b64_json'

    @model_validator(mode='after')
    def options(self):
        if self.model != MODEL_ID:
            raise ValueError(f'model must be {MODEL_ID}')
        if self.response_format != 'b64_json':
            raise ValueError('Only response_format=b64_json is supported.')
        if self.size:
            try:
                self.width, self.height = map(int, self.size.lower().split('x'))
            except (ValueError, TypeError):
                raise ValueError('size must be WIDTHxHEIGHT')
            if not (512 <= self.width <= 2752 and 512 <= self.height <= 2752):
                raise ValueError('Each dimension must be between 512 and 2752.')
            self.dimensions()
        return self


@app.get('/health')
async def health():
    if pipeline is None:
        raise HTTPException(503, 'Model not loaded')
    return {'status': 'ready', 'model': MODEL_ID, 'loaded_at': loaded_at, 'busy': lock.locked()}


@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model', 'owned_by': 'Qwen'}]}


def infer(req, seed):
    import torch
    started = time.monotonic()
    try:
        with torch.inference_mode():
            result = pipeline(
                prompt=req.prompt, width=req.width, height=req.height,
                num_inference_steps=req.steps,
                generator=torch.Generator('cuda').manual_seed(seed),
            ).images[0]
        buffer = BytesIO()
        result.save(buffer, format='PNG')
        elapsed = time.monotonic() - started
        log.info('Generated %sx%s, %s steps, seed %s in %.2fs; peak CUDA allocated %.2f GiB',
                 req.width, req.height, req.steps, seed, elapsed,
                 torch.cuda.max_memory_allocated() / 1024**3)
        return buffer.getvalue(), elapsed
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(503, 'GPU memory exhausted; try a smaller image or free RAM.')


async def generate(req):
    if pipeline is None:
        raise HTTPException(503, 'Model not loaded')
    # No await between checking/acquiring: only one request uses shared pipeline state.
    if lock.locked():
        raise HTTPException(429, 'Generation in progress; retry after it completes.')
    async with lock:
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        # Keep the lock until the GPU work finishes even if a client disconnects.
        task = asyncio.create_task(asyncio.to_thread(infer, req, seed))
        cancelled = False
        while True:
            try:
                png, elapsed = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():  # Event-loop shutdown can cancel the worker too.
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return png, seed, elapsed


@app.post('/generate')
async def generate_png(req: GenerateRequest):
    png, seed, elapsed = await generate(req)
    return Response(png, media_type='image/png', headers={
        'X-Seed': str(seed), 'X-Generation-Seconds': f'{elapsed:.3f}',
    })


@app.post('/v1/images/generations')
async def generate_json(req: OpenAIRequest):
    png, seed, elapsed = await generate(req)
    return {'created': int(time.time()), 'data': [{'b64_json': base64.b64encode(png).decode()}],
            'seed': seed, 'generation_seconds': elapsed}
