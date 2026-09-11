import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from vulnmechanism.process import run_process, terminate_process_group


class ProcessTests(unittest.TestCase):
    def test_keyboard_interrupt_cleans_up_and_propagates(self):
        process = MagicMock()
        process.communicate.side_effect = [KeyboardInterrupt, ('', '')]
        with patch('vulnmechanism.process.subprocess.Popen', return_value=process), \
                patch('vulnmechanism.process.terminate_process_group') as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                run_process(['unused'], timeout=1)
            cleanup.assert_called_once_with(process)

    def test_exited_parent_does_not_skip_group_cleanup(self):
        process = MagicMock(pid=123456)
        process.poll.return_value = 0
        with patch('vulnmechanism.process.os.killpg') as kill:
            terminate_process_group(process)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(123456, signal.SIGTERM), (123456, signal.SIGKILL)])

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires POSIX process groups')
    def test_timeout_kills_descendant_after_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / 'child.pid'
            program = '''import os, signal, sys, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    open(sys.argv[1], 'w').write(str(os.getpid()))
    time.sleep(60)
else:
    os._exit(0)
'''
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, '-c', program, str(pid_path)], timeout=1)
                pid = int(pid_path.read_text())
                stat = Path(f'/proc/{pid}/stat')
                # An orphan may remain as a zombie until PID 1 reaps it.
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')
            finally:
                if pid_path.exists():
                    try:
                        os.kill(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
