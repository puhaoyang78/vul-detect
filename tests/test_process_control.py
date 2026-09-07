import os
import subprocess
import sys
import time
import unittest

from semantic_demo.joern import _run_process_group


@unittest.skipUnless(hasattr(os, "killpg"), "process-group test requires POSIX")
class ProcessGroupTests(unittest.TestCase):
    def test_timeout_terminates_spawned_child_process(self):
        command = [
            sys.executable,
            "-c",
            (
                "import subprocess,time; "
                "p=subprocess.Popen(['sleep','30']); "
                "print(p.pid, flush=True); "
                "time.sleep(30)"
            ),
        ]
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            _run_process_group(command, timeout=0.5)
        output = caught.exception.output or ""
        child_pid = int(output.strip().splitlines()[0])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail(f"timed-out child process {child_pid} is still alive")


if __name__ == "__main__":
    unittest.main()
