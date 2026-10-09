"""Tests for the release changelog scripts in .github/scripts: release_version.py and update_changelog.py."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"

# Older releases with Markdown that must survive untouched: a hard line break
# (two trailing spaces, written as "\x20\x20" so editors and the pre-commit
# trailing-whitespace hook don't strip them) and blank lines in a code block.
HISTORY = (
    "# Version  0.2.0\n"
    "\n"
    "#### New Features\n"
    "\n"
    "* [#477](https://github.com/pykale/pykale/pull/477): Add Integrated Gradients\x20\x20\n"
    "  with a hard line break\n"
    "\n"
    "```python\n"
    "x = 1\n"
    "\n"
    "\n"
    "y = 2\n"
    "```\n"
    "\n"
    "# Version  0.1.2\n"
    "\n"
    "#### Bug Fixes\n"
    "\n"
    "* [#389](https://github.com/pykale/pykale/pull/389): Fix something\n"
)

GENERATED = """#### New Features

* [#590](https://github.com/pykale/pykale/pull/590): Add X

#### Bug Fixes


#### Other Changes

* [#591](https://github.com/pykale/pykale/pull/591): Tidy Y
"""


def load(name):
    """Loads a script from .github/scripts as a module, since that folder is not a package.

    Args:
        name (str): The script's file name without the ".py" extension.

    Returns:
        module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_version = load("release_version")
update_changelog = load("update_changelog")


# release_version.py


@pytest.mark.parametrize(
    "title, package_version, expected",
    [
        ("Release 0.3.0", "0.3.0", "0.3.0"),
        ("Prepare the 0.3.0 release", "0.3.0", "0.3.0"),
        ("RELEASE v0.3.0.", "0.3.0", "0.3.0"),
        ("Release (0.3.0)", "0.3.0", "0.3.0"),
        ("Release 0.3.0a1", "0.3.0a1", "0.3.0a1"),
        ("Release 0.3.0b2", "0.3.0b2", "0.3.0b2"),
        ("Release 0.3.0RC1", "0.3.0rc1", "0.3.0rc1"),
    ],
)
def test_release_version_accepted(title, package_version, expected):
    """Tests that plain releases and a/b/rc pre-releases matching the package version are accepted."""
    assert release_version.find_release_version(title, package_version) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Add MOGONET example",
        "Fix prerelease check in 0.3.0",
        "Update released notebooks for 0.3.0",
        "Fix release publishing workflow",
        "Release use ubuntu-latest",
    ],
)
def test_release_version_not_a_release(title):
    """Tests that titles without the word "release" or without a version are not treated as releases."""
    assert release_version.find_release_version(title, "0.3.0") is None


@pytest.mark.parametrize(
    "title, message",
    [
        ("Release 0.3.0-beta.1", "Unsupported version '0.3.0-beta.1'"),
        ("Release 0.3.0.post1", "Unsupported version '0.3.0.post1'"),
        ("Release 0.3.0dev1", "Unsupported version '0.3.0dev1'"),
        ("Release notes for 0.2.0 and 0.3.0", "exactly one version, found: 0.2.0 0.3.0"),
        ("Release 0.3.1", "does not match kale/__init__.py (0.3.0)"),
        # A truncated version must be reported, not mistaken for a title that merely mentions
        # releases: that let a release pull request pass with no changelog generated.
        ("Release 0.3", "Unsupported version '0.3'"),
        ("Release v0.3", "Unsupported version 'v0.3'"),
        ("Release 0.3 and 0.4", "exactly one version, found: 0.3 0.4"),
    ],
)
def test_release_version_rejected(title, message):
    """Tests that unsupported, ambiguous and mismatched versions raise an error saying what to fix."""
    with pytest.raises(release_version.ReleaseTitleError, match=message.replace("(", r"\(").replace(")", r"\)")):
        release_version.find_release_version(title, "0.3.0")


def test_release_version_reads_package_version(tmp_path):
    """Tests reading ``__version__`` from ``__init__.py``, and the error when it is missing."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text('"""Docs."""\n__version__ = "0.3.0rc1"\n', encoding="utf-8")
    assert release_version.read_package_version(init_file) == "0.3.0rc1"

    init_file.write_text("VERSION = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        release_version.read_package_version(init_file)


@pytest.mark.parametrize(
    "title, exit_code, output",
    [
        ("Release 0.3.0", 0, "version=0.3.0\n"),
        ("Add MOGONET example", 0, "version=\n"),
        ("Release 0.3.1", 1, ""),
    ],
)
def test_release_version_main(tmp_path, monkeypatch, capsys, title, exit_code, output):
    """Tests the command-line interface's exit code, ``$GITHUB_OUTPUT`` and error annotation."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    github_output = tmp_path / "github_output"
    github_output.touch()
    monkeypatch.setenv("PR_TITLE", title)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr("sys.argv", ["release_version.py", str(init_file)])

    assert release_version.main() == exit_code
    assert github_output.read_text(encoding="utf-8") == output
    if exit_code:
        assert capsys.readouterr().out.startswith("::error::")


# update_changelog.py


def run_updater(monkeypatch, version, generated, changelog):
    """Runs update_changelog.py's command-line interface with the given arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): The fixture used to set ``sys.argv``.
        version (str): The release version argument.
        generated (Path): The file generated by the action, rewritten to hold the comment body.
        changelog (Path): The changelog file to update.

    Returns:
        int: The exit code returned by ``main()``.
    """
    monkeypatch.setattr("sys.argv", ["update_changelog.py", version, str(generated), str(changelog)])
    return update_changelog.main()


def test_drop_empty_groups():
    """Tests that group headings without entries are removed and the others kept."""
    body = update_changelog.drop_empty_groups(GENERATED)
    assert "#### Bug Fixes" not in body
    assert body.startswith("#### New Features")
    assert body.endswith("Tidy Y")


def test_insert_keeps_history_byte_for_byte(tmp_path, monkeypatch):
    """Tests that a new section is inserted at the top and older releases are left byte for byte."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())
    generated = tmp_path / "generated.md"
    generated.write_text(GENERATED, encoding="utf-8")

    assert run_updater(monkeypatch, "v0.3.0", generated, changelog) == 0

    content = changelog.read_bytes().decode()
    section, rest = content.split("\n\n# Version  0.2.0", 1)
    assert section.startswith("# Version  0.3.0\n\n#### New Features\n\n* [#590]")
    assert "#### Bug Fixes" not in section
    # Trailing spaces and blank lines in older releases are kept unchanged.
    assert "# Version  0.2.0" + rest == HISTORY
    # The comment body is the new section.
    assert generated.read_text(encoding="utf-8") == section + "\n"


def test_existing_section_is_never_overwritten(tmp_path, monkeypatch):
    """Tests that a curated section for the version is kept and the fresh list only goes to the comment."""
    curated = "# Version  0.3.0\n\n#### New Features\n\n* Reworded by a maintainer\n\n" + HISTORY
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(curated.encode())
    generated = tmp_path / "generated.md"
    generated.write_text(GENERATED, encoding="utf-8")

    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0

    assert changelog.read_bytes().decode() == curated
    comment = generated.read_text(encoding="utf-8")
    assert "already has a section for version 0.3.0" in comment
    assert "* [#590](https://github.com/pykale/pykale/pull/590): Add X" in comment


def test_all_groups_empty_writes_placeholder(tmp_path, monkeypatch):
    """Tests that a placeholder entry is written when every group is empty."""
    changelog = tmp_path / "CHANGELOG.md"
    generated = tmp_path / "generated.md"
    generated.write_text("#### Other Changes\n\n", encoding="utf-8")

    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0
    assert changelog.read_text(encoding="utf-8") == "# Version  0.3.0\n\n* No changelog entries for this release.\n"


def test_missing_generated_file_fails(tmp_path, monkeypatch):
    """Tests that a missing generated file fails without touching the changelog."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())

    assert run_updater(monkeypatch, "0.3.0", tmp_path / "missing.md", changelog) == 1
    assert changelog.read_bytes().decode() == HISTORY


# Retrying the workflow after the changelog commit was already pushed


def git(repo, *args):
    """Runs a git command in a repository.

    Args:
        repo (Path): The repository directory.
        *args (str): The git subcommand and its arguments.

    Returns:
        str: The command's standard output, stripped.
    """
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_rerun_from_branch_tip_needs_no_second_commit(tmp_path, monkeypatch):
    """Tests a rerun whose first attempt pushed the changelog but failed to comment.

    The workflow checks out the branch tip, so the rerun sees the pushed
    section, changes nothing (no second commit that would be rejected as
    non-fast-forward) and still writes the comment body.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "release-0.3.0")
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "core.autocrlf", "false")
    changelog = repo / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())
    git(repo, "add", "CHANGELOG.md")
    git(repo, "commit", "-q", "-m", "Bump version to 0.3.0")
    event_sha = git(repo, "rev-parse", "HEAD")
    generated = tmp_path / "generated.md"

    # First attempt: inserts the section and pushes it, then fails to comment.
    generated.write_text(GENERATED, encoding="utf-8")
    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0
    git(repo, "commit", "-q", "-am", "Update CHANGELOG.md for version 0.3.0")
    pushed_sha = git(repo, "rev-parse", "HEAD")
    assert pushed_sha != event_sha

    # Rerun: checks out the branch tip, which already holds the section.
    git(repo, "checkout", "-q", "release-0.3.0")
    generated.write_text(GENERATED, encoding="utf-8")
    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0

    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "rev-parse", "HEAD") == pushed_sha
    assert "* [#590](https://github.com/pykale/pykale/pull/590): Add X" in generated.read_text(encoding="utf-8")
