# Qwen Image 2.1 on DGX Spark

Persistent local BF16 image generation server. Weights load once at startup and
stay on the GPU/shared memory for subsequent requests. Requests are serialized;
a second concurrent generation receives HTTP 429 rather than consuming extra RAM.

## Set up this repository

Supported target: **NVIDIA DGX Spark running Linux ARM64**. The lockfile is
deliberately scoped to that platform; this is not a tested setup for x86-64 PCs,
macOS, Windows, or CPU-only inference.

Before installing:

- Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
  uv can download the pinned Python 3.14.2 interpreter automatically.
- Check `nvidia-smi`. The pinned PyTorch build uses CUDA 13.0 and needs a
  compatible NVIDIA driver from the R580 branch or later; see
  [NVIDIA's compatibility table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver).
- Plan for **at least 60 GiB of free disk space** for the approximately 31 GiB
  model, Python/CUDA packages, download caches, and headroom. This is a planning
  allowance, not a measured minimum. Check both the checkout and uv cache
  filesystem (`uv cache dir`) if they are on different disks.
- The default launcher requires **56 GiB of available shared RAM**. CPU offload
  does not create additional physical memory on Spark.
- A Linux user systemd session is needed for background operation. Foreground
  mode works without it. Docker and Tailscale are optional.

```sh
git clone https://github.com/anthonywu/dgx-serve-qwen21.git
cd dgx-serve-qwen21
uv sync
```

`uv sync` creates `.venv/` using the interpreter pinned in `.python-version`
and installs the dependencies from `uv.lock`. The project targets Linux ARM64
on DGX Spark. Use `uv sync --locked` to require an unchanged lockfile.

Model weights, the virtual environment, logs, generated outputs, and local
verification receipts are excluded from Git. The sample gallery is included.
No pre-existing model cache, local verification receipt, or virtual environment
is required. Run the download step below before starting the server.

## Download and verification

Model: `Qwen/Qwen-Image-2.1`, revision
`b3179ad355be050328e483a9dfdd9e60cd62adfa` (about 30.86 GiB).

```sh
uv run --no-sync download.py
```

Downloads resume from `model/`. Every file is checked for size and upstream hash
(SHA-256 for weight files, Git blob SHA-1 for ordinary repository files).
Success creates `verified.json`. Startup checks that those verified files have
not changed. Rerun the downloader for a complete checksum recheck.

The downloader uses Hugging Face's default transfer settings and respects its
environment variables. If an unstable connection causes repeated stalls, try
`HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8 uv run --no-sync download.py`; this is an
optional tuning choice, not a requirement for every network.

Inspect local download and server status with `uv run --no-sync status.py`.

## Start

```sh
uv run --no-sync start.py --check   # preview RAM and recognized inference servers, no changes
uv run --no-sync start.py
```

Without a user systemd session, run in the current terminal instead:

```sh
uv run --no-sync start.py --foreground
```

Wait for Uvicorn's `Application startup complete` message in foreground mode.
Ctrl-C stops the foreground server. Model loading can take several minutes.
Keep one worker: each additional worker would load another copy of the model.

Default endpoint: **http://127.0.0.1:8021**. Interactive API docs: `/docs`.
The launcher checks verified weights, pipeline imports, a CUDA BF16 computation,
port availability, and the user service manager before offering to stop anything.

The default RAM budget is **56 GiB available** (`MemAvailable`, excluding swap).
This is a conservative startup guard, not a measured peak for every resolution.
If sufficient RAM is already free, it starts without stopping existing servers.
It does not require Docker access in that case. `--check` and low-memory handoff
inspect Docker when its CLI is installed and require access to its daemon;
inspection errors abort rather than risk stopping unidentified workloads.
Otherwise it lists recognized inference containers/processes and asks:

```
Stop these inference servers and start Qwen Image 2.1? [Y/n]
```

Enter / `y` / `yes` approve; `n` / `no`, EOF, or Ctrl-C at the prompt cancel.
There is no automatic yes flag. The launcher only stops the listed containers
and matching host processes, waits for RAM to become available, and refuses
startup if processes respawn or RAM is still insufficient. Docker stop uses a
60-second grace period; host processes receive SIGTERM with no forced kill.
Recognizes SGLang, vLLM, llama.cpp, Ollama, TGI, and Aphrodite. Other workloads
must be managed separately. It does not automatically restart stopped servers.

The server runs as a user systemd service and survives closing the launcher.
After printing `Ready`, the launcher intentionally exits and returns to your
shell; that does not stop the model server.
It does not automatically start on boot. Full logout can stop user services
unless systemd lingering is enabled for the account. The launcher waits up to
15 minutes for `/health` to report that weights are loaded. Startup errors go to
the journal.

```sh
curl --fail http://127.0.0.1:8021/health
curl --fail http://127.0.0.1:8021/v1/models
journalctl --user -u qwen21-server.service -f
systemctl --user stop qwen21-server.service
```

The background unit name is shared by checkouts under the same Linux account.
Use foreground mode on a different port for an additional instance, with enough
RAM for another copy of the model. The status and sample scripts accept the
same base URL as the smoke test:

```sh
uv run --no-sync start.py --foreground --port 8022
# In another terminal:
uv run --no-sync status.py --url http://127.0.0.1:8022
uv run --no-sync smoke_test.py --url http://127.0.0.1:8022
uv run --no-sync generate_samples.py --url http://127.0.0.1:8022
```

## Generate an image

```sh
curl --fail-with-body http://127.0.0.1:8021/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red panda painting in a sunlit studio","width":1024,"height":1024,"steps":40,"seed":42}' \
  --output image.png
```

`/generate` returns PNG bytes, with `X-Seed` and `X-Generation-Seconds` headers.
The model remains loaded after the request. No model files are downloaded at
server startup. The default is 1024x1024; 2048x2048 is also accepted. Dimensions
must be multiples of 32, each between 512 and 2752, with at most 4.4 million pixels.
Steps range from 1 to 100. Lower step counts are useful for smoke tests but reduce
quality. For transparency, include the model's recommended phrasing:
`This is an RGBA image with transparency. ... The image has alpha channel and the background is transparent.`

An OpenAI-style subset is also supported:

```sh
curl --fail-with-body http://127.0.0.1:8021/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image-2.1","prompt":"A red panda painting","size":"1024x1024","n":1,"response_format":"b64_json"}'
```

This endpoint returns `data[0].b64_json`. It supports one image per request and
base64 output only. This server currently exposes text-to-image generation;
image-editing inputs are not implemented. Bind address and port can be changed
with `--host` and `--port`. The default loopback binding is appropriate for this
unauthenticated local API.

For a live smoke test (two 512x512, two-step generations with the same loaded
model), run `uv run --no-sync smoke_test.py`. Test images are saved in `outputs/`.

## Runtime and tests

Python 3.14.2 virtual environment in `.venv/`. `.python-version` pins the
interpreter, `pyproject.toml` declares dependencies, and `uv.lock` records the
validated package set. The default sync includes the test dependencies.

```sh
uv sync --locked
uv run --no-sync python -m unittest -v test_start.py test_server.py
```

The pinned NVIDIA cuSPARSELt 0.8.1 ARM64 wheel reports an SBSA platform tag
internally. `uv pip check` and `uv sync --check` therefore report a compatibility
issue, and `uv sync` may reinstall that package. CUDA execution is tested
separately; the wheel metadata is not modified. After syncing, use
`uv run --no-sync` to run commands without changing the environment.

When updating an existing installation, finish any active image request and
stop the service before syncing its environment, then restart it:

```sh
systemctl --user stop qwen21-server.service
uv sync --locked
uv run --no-sync start.py
```

Upstream model and license: https://huggingface.co/Qwen/Qwen-Image-2.1

## Optional Tailscale access

The server's `/` page provides a sample gallery and a form for generating images.
Sample PNGs and their prompts live under `samples/`; regenerate them with:

```sh
uv run --no-sync generate_samples.py
```

The gallery also works locally at `http://127.0.0.1:8021`; Tailscale is not
required. To share it with your tailnet, first install Tailscale, sign the machine
in, and enable HTTPS for your tailnet. Then expose the local server on port 8021:

```sh
tailscale serve --bg --https=8021 http://127.0.0.1:8021
tailscale serve status
```

Use the HTTPS URL reported by `tailscale serve status`. The same origin provides
`/generate`, `/v1/images/generations`, `/health`, and `/docs`. Only clients allowed
by your Tailscale network policy can reach this Serve endpoint. Choose an unused
Serve port and preserve any existing mappings.

To remove only this mapping:

```sh
tailscale serve --https=8021 off
```

The Serve mapping persists until removed. Start the model with `uv run --no-sync start.py`
whenever it is not running; the URL only works while the model server is up.

## License

The code in this repository is licensed under the [MIT License](LICENSE).
Model weights and third-party dependencies retain their respective upstream
licenses.
