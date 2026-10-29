"""Prevent accidental publication of local coursework and private materials."""

from pathlib import Path
import tempfile
import unittest

from scripts.verify_project import AuditError, audit_files, audit_ignore_rules


class RepositoryPrivacyTests(unittest.TestCase):
    def test_required_directories_and_credentials_are_ignored(self):
        audit_ignore_rules(Path(__file__).resolve().parents[1])

    def test_forced_private_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("assignment/task.md", "reports/report.md", "data/photo.png",
                         "results/metrics.json", "private/notes.md", "scratch/output.txt"):
                relative = Path(name)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("local material", encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(AuditError):
                    audit_files(root, [relative])

    def test_readme_cannot_link_to_ignored_local_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            (root / "reports/report.md").write_text("local report", encoding="utf-8")
            (root / "README.md").write_text("[Report](reports/report.md)", encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "local-only"):
                audit_files(root, [Path("README.md")])


if __name__ == "__main__":
    unittest.main()
