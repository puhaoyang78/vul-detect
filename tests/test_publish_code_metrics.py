from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublishCodeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="publish-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.remote = Path(self.temp.name) / "origin.git"
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1",
                        GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.env.pop(key, None)
        self.git("init", "--quiet")
        self.git("symbolic-ref", "HEAD", "refs/heads/main")
        self.git("config", "user.name", "Publish test")
        self.git("config", "user.email", "publish-test@example.invalid")
        self.write("README.md", "Initial code.\n")
        self.write("docs/obsolete.md", "Old documentation.\n")
        self.write("scripts/publish_code_metrics.sh",
                   (ROOT / "scripts/publish_code_metrics.sh").read_text())
        shutil.copyfile(ROOT / ".gitignore", self.repo / ".gitignore")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Fixture base")
        self.git("init", "--bare", "--quiet", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "--quiet", "origin", "HEAD:main")
        self.initial_head = self.git("rev-parse", "HEAD").stdout.strip()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, env=self.env,
                              text=True, capture_output=True, check=True, timeout=15)

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def publish(self, *args):
        return subprocess.run(["bash", "scripts/publish_code_metrics.sh", *args],
                              cwd=self.repo, env=self.env, text=True,
                              capture_output=True, timeout=15)

    def assert_unpublished(self):
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.initial_head)
        remote_head = self.git("--git-dir", str(self.remote), "rev-parse", "refs/heads/main")
        self.assertEqual(remote_head.stdout.strip(), self.initial_head)

    def test_dry_run_detects_untracked_whitespace_without_staging(self):
        self.write("tests/example.py", "value = 1 \n")
        index = (self.repo / ".git/index").read_bytes()
        result = self.publish("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tests/example.py:1: trailing whitespace", result.stdout + result.stderr)
        self.assertIn("Whitespace check failed", result.stderr)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assert_unpublished()

    def test_whitespace_failure_preserves_existing_staged_changes(self):
        self.write("README.md", "Already staged.\n")
        self.git("add", "README.md")
        self.write("README.md", "Working version with trailing spaces. \n")
        index = (self.repo / ".git/index").read_bytes()
        staged_diff = self.git("diff", "--cached").stdout
        result = self.publish("Should fail preflight")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Whitespace check failed", result.stderr)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assertEqual(self.git("diff", "--cached").stdout, staged_diff)
        self.assert_unpublished()

    def test_valid_dry_run_does_not_change_index_or_remote(self):
        self.write("README.md", "Updated code.\n")
        self.write("results/run/test.metrics.json", '{"f1": 0.5}\n')
        self.write("results/run/best.pt", "Local checkpoint.\n")
        index = (self.repo / ".git/index").read_bytes()
        result = self.publish("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("results/run/test.metrics.json", result.stdout)
        self.assertNotIn("best.pt", result.stdout)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assert_unpublished()

    def test_publishes_code_metrics_and_unstaged_deletions(self):
        self.write("README.md", "Updated code.\n")
        self.write("tests/new test.py", "value = 1\n")
        self.write("results/run/test.metrics.json", '{"f1": 0.5}\n')
        self.write("results/run/best.training.jsonl", '{"epoch": 1}\n')
        self.write("results/run/best.pt", "Local checkpoint.\n")
        self.write("results/run/predictions.jsonl", '{"prediction": 1}\n')
        self.write("data/functions.jsonl", '{"source": "local"}\n')
        (self.repo / "docs/obsolete.md").unlink()
        result = self.publish("Publish fixture changes")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        committed = set(self.git("diff", "--name-only", self.initial_head, "HEAD").stdout.splitlines())
        self.assertEqual(committed, {"README.md", "docs/obsolete.md", "tests/new test.py",
                                     "results/run/test.metrics.json", "results/run/best.training.jsonl"})
        remote_head = self.git("--git-dir", str(self.remote), "rev-parse", "refs/heads/main")
        self.assertEqual(remote_head.stdout, self.git("rev-parse", "HEAD").stdout)

    def test_staged_out_of_scope_file_is_still_rejected(self):
        self.write("data/functions.jsonl", '{"source": "local"}\n')
        self.git("add", "--force", "data/functions.jsonl")
        index = (self.repo / ".git/index").read_bytes()
        result = self.publish("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside code/metric scope", result.stderr)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assert_unpublished()


if __name__ == "__main__":
    unittest.main()
