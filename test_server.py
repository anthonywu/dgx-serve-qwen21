import asyncio
import base64
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pydantic import ValidationError
import server

PNG = b'\x89PNG\r\n\x1a\nmock'


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)  # No context manager: do not load real weights.
        self.old_pipeline = server.pipeline
        server.pipeline = object()
        server.lock = asyncio.Lock()

    def tearDown(self):
        server.pipeline = self.old_pipeline
        self.client.close()

    def test_health_requires_loaded_weights(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        server.pipeline = None
        self.assertEqual(self.client.get('/health').status_code, 503)

    def test_png_and_json_outputs(self):
        with patch.object(server, 'infer', return_value=(PNG, 1.2)) as infer:
            response = self.client.post('/generate', json={'prompt':'test', 'seed':42})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, PNG)
            self.assertEqual(response.headers['x-seed'], '42')
            response = self.client.post('/v1/images/generations', json={
                'prompt':'test', 'size':'1536x1024', 'seed':1})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(base64.b64decode(response.json()['data'][0]['b64_json']), PNG)
            self.assertEqual(infer.call_args.args[0].width, 1536)

    def test_invalid_requests_never_reach_gpu(self):
        with patch.object(server, 'infer') as infer:
            for payload in ({'prompt':''}, {'prompt':'x','width':513},
                            {'prompt':'x','width':2752,'height':2752},
                            {'prompt':'x','steps':0}):
                self.assertEqual(self.client.post('/generate',json=payload).status_code,422)
            for payload in ({'size':'99999x99999'}, {'size':'foo'}, {'n':2},
                            {'response_format':'url'}, {'model':'wrong'}):
                self.assertEqual(self.client.post('/v1/images/generations',
                                 json={'prompt':'x',**payload}).status_code,422)
            infer.assert_not_called()


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_rejected(self):
        server.lock = asyncio.Lock()
        with patch.object(server, 'pipeline', object()):
            async with server.lock:
                with self.assertRaises(server.HTTPException) as caught:
                    await server.generate(server.GenerateRequest(prompt='test'))
                self.assertEqual(caught.exception.status_code, 429)

    async def test_cancellation_holds_lock_until_inference_finishes(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        def infer(*args):
            entered.set()
            release.wait(5)
            return PNG, 0.1
        server.lock = asyncio.Lock()
        with patch.object(server, 'pipeline', object()), patch.object(server, 'infer', infer):
            task = asyncio.create_task(server.generate(server.GenerateRequest(prompt='test')))
            await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertTrue(server.lock.locked())
            task.cancel()  # A second cancellation must not cancel the worker future.
            await asyncio.sleep(0.01)
            self.assertTrue(server.lock.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(server.lock.locked())


if __name__ == '__main__':
    unittest.main()
