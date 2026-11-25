"""CPU-only switch failure-path regression."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class SwitchFailureTest(unittest.TestCase):
    def test_send_and_socket_statistics_failure_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'test_switch'
            subprocess.run([*shlex.split(os.environ.get('CC', 'cc')), '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            str(Path(__file__).with_suffix('.c')), '-o', str(binary)], check=True, capture_output=True)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS:', result.stdout)


if __name__ == '__main__': unittest.main()
