import unittest
from types import SimpleNamespace
from unittest.mock import patch
import start


class ConsentTests(unittest.TestCase):
    def test_enter_and_explicit_yes_approve(self):
        for answer in ('', 'y', 'Y', 'yes'):
            with patch('builtins.input', return_value=answer):
                self.assertTrue(start.confirm())

    def test_no_and_eof_cancel(self):
        for answer in ('n', 'N', 'no'):
            with patch('builtins.input', return_value=answer):
                self.assertFalse(start.confirm())
        with patch('builtins.input', side_effect=EOFError):
            self.assertFalse(start.confirm())

    def test_cancel_has_no_stop_or_start_side_effects(self):
        for answer in ('n', EOFError):
            with patch('sys.argv', ['start.py']), \
                 patch.object(start.shutil, 'which', return_value='/bin/systemctl'), \
                 patch.object(start, 'available_gib', return_value=8), \
                 patch.object(start, 'discover', return_value=([{'id':'abc','name':'example-inference'}], [])), \
                 patch.object(start, 'preflight'), \
                 patch.object(start, 'stop_targets') as stop, \
                 patch.object(start.subprocess, 'run') as run, \
                 patch('builtins.input', **({'side_effect':EOFError} if answer is EOFError else {'return_value':answer})):
                run.return_value.returncode = 3
                self.assertEqual(start.main(), 0)
                stop.assert_not_called()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][:3], ['systemctl', '--user', 'is-active'])

    def test_pid_reuse_is_not_signalled(self):
        with patch.object(start, 'identity', return_value='new'), \
             patch.object(start, 'discover', return_value=([], [])), \
             patch.object(start.os, 'kill') as kill:
            start.stop_targets([], [{'pid':123, 'identity':'old'}])
            kill.assert_not_called()

    def test_process_detection(self):
        for cmd in ('python -m sglang.launch_server --model x', 'sglang::scheduler',
                    'vllm serve model', '/bin/ollama serve', '/opt/llama-server -m file'):
            self.assertIsNotNone(start.SERVER.search(cmd), cmd)
        for cmd in ('redis-server', 'python server.py', 'bash echo vllm-model', '/usr/bin/gnome-shell'):
            self.assertIsNone(start.SERVER.search(cmd), cmd)


class PortableLaunchTests(unittest.TestCase):
    def test_active_service_from_another_checkout_is_not_reported_as_ours(self):
        with patch('sys.argv', ['start.py']), \
             patch.object(start.shutil, 'which', return_value='/bin/systemctl'), \
             patch.object(start.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(start, 'run', return_value='/different/checkout'), \
             patch.object(start, 'ready') as ready:
            with self.assertRaisesRegex(RuntimeError, 'another checkout'):
                start.main()
        ready.assert_not_called()

    def test_foreground_with_enough_ram_needs_neither_docker_nor_systemd(self):
        with patch('sys.argv', ['start.py', '--foreground', '--port', '8022']), \
             patch.object(start, 'available_gib', return_value=64), \
             patch.object(start, 'discover', side_effect=AssertionError('Docker inspection not needed')), \
             patch.object(start.shutil, 'which', side_effect=AssertionError('No systemd required')), \
             patch.object(start, 'preflight') as preflight, \
             patch.object(start.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(start.main(), 0)
        preflight.assert_called_once_with('127.0.0.1', 8022, foreground=True)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [str(start.PYTHON), '-m', 'uvicorn', 'server:app'])
        self.assertEqual(command[command.index('--port') + 1], '8022')
        self.assertEqual(run.call_args.kwargs['cwd'], start.ROOT)
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')

    def test_background_with_enough_ram_does_not_inspect_docker(self):
        with patch('sys.argv', ['start.py']), \
             patch.object(start.shutil, 'which', return_value='/bin/systemctl'), \
             patch.object(start, 'available_gib', return_value=64), \
             patch.object(start, 'discover', side_effect=AssertionError('Docker inspection not needed')), \
             patch.object(start, 'preflight'), \
             patch.object(start, 'ready', return_value=True), \
             patch.object(start.subprocess, 'run', return_value=SimpleNamespace(returncode=3)) as run:
            self.assertEqual(start.main(), 0)
        self.assertTrue(any(call.args[0][0] == 'systemd-run' for call in run.call_args_list))

    def test_foreground_handoff_cancel_never_starts_or_stops_a_process(self):
        with patch('sys.argv', ['start.py', '--foreground']), \
             patch.object(start, 'available_gib', return_value=8), \
             patch.object(start, 'discover', return_value=([{'id':'abc','name':'example-inference'}], [])), \
             patch.object(start, 'preflight'), \
             patch.object(start, 'stop_targets') as stop, \
             patch.object(start.subprocess, 'run') as run, \
             patch('builtins.input', return_value='n'):
            self.assertEqual(start.main(), 0)
        stop.assert_not_called()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
