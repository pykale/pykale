#!/usr/bin/env python3
"""Read the release version from a pull request title.

Usage: PR_TITLE="Release 0.3.0" release_version.py [init-file]

A pull request is a release when its title contains the whole word "release"
and a version. The version must be a single plain release or a pre-release
that pykale publishes (0.3.0, 0.3.0a1, 0.3.0b1 or 0.3.0rc1), and it must match
``__version__`` in ``kale/__init__.py``, which is bumped before the release pull
request is opened.

The version is written to ``$GITHUB_OUTPUT`` as ``version=<version>``, or as
``version=`` when the pull request is not a release. A title that looks like a
release but carries a wrong version exits with an error saying what to fix.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

RELEASE_RE = re.compile(r"(^|[^a-z])release([^a-z]|$)", re.IGNORECASE)
CANDIDATE_RE = re.compile(r"^[vV]?[0-9]+\.[0-9]+\.[0-9]+")
SUPPORTED_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$")
PACKAGE_VERSION_RE = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
# Surrounding brackets and quotes, and trailing punctuation such as a full stop.
LEADING_PUNCTUATION = "([\"'"
TRAILING_PUNCTUATION = "]).,;:!?\"'"


class ReleaseTitleError(ValueError):
    """Raised when a pull request title looks like a release but its version can't be used.

    The message says what to fix, e.g. an unsupported version format or a mismatch with ``kale/__init__.py``.
    """


def find_release_version(title: str, package_version: str) -> str | None:
    """Finds the release version in a pull request title.

    A title is a release when it contains the whole word "release" (so "prerelease" and "released" don't count) and
    a word starting with X.Y.Z. Surrounding brackets, quotes and trailing punctuation are ignored.

    Args:
        title (str): The pull request title, e.g. "Release 0.3.0".
        package_version (str): The version in ``kale/__init__.py``, which the title must match.

    Returns:
        str or None: The normalised version, e.g. "0.3.0rc1", or None if the title is not a release.

    Raises:
        ReleaseTitleError: If the title carries more than one version, a version other than X.Y.Z, X.Y.ZaN, X.Y.ZbN
            or X.Y.ZrcN, or a version that differs from ``package_version``.
    """
    if not RELEASE_RE.search(title):
        return None

    # Keep any suffix attached, so "0.3.0-beta.1" and "0.3.0.post1" are checked
    # whole instead of being cut down to "0.3.0".
    words = (word.lstrip(LEADING_PUNCTUATION).rstrip(TRAILING_PUNCTUATION) for word in title.split())
    candidates = [word for word in words if CANDIDATE_RE.match(word)]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ReleaseTitleError(f"Release title must contain exactly one version, found: {' '.join(candidates)}")

    version = candidates[0].lower().removeprefix("v")
    if not SUPPORTED_RE.match(version):
        raise ReleaseTitleError(
            f"Unsupported version '{candidates[0]}'. Use X.Y.Z or a pre-release such as X.Y.Za1, X.Y.Zb1 or X.Y.Zrc1."
        )
    if version != package_version:
        raise ReleaseTitleError(
            f"Title version {version} does not match kale/__init__.py ({package_version}). "
            "Bump the version first, or fix the title."
        )
    return version


def read_package_version(init_file: Path) -> str:
    """Reads ``__version__`` from a package's ``__init__.py`` the same way setup.py does.

    Args:
        init_file (Path): Path to the ``__init__.py`` file, e.g. ``kale/__init__.py``.

    Returns:
        str: The package version, e.g. "0.3.0".

    Raises:
        RuntimeError: If the file does not define ``__version__``.
    """
    match = PACKAGE_VERSION_RE.search(init_file.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Unable to find __version__ in {init_file}")
    return match.group(1)


def main() -> int:
    """Runs the command-line interface: validates ``$PR_TITLE`` and writes the version to ``$GITHUB_OUTPUT``.

    Takes an optional path to the package's ``__init__.py`` as its only argument, defaulting to
    ``kale/__init__.py``. Messages use GitHub workflow commands (``::notice::`` and ``::error::``).

    Returns:
        int: The exit code, 0 for a valid release or a non-release title, or 1 for an invalid release title.
    """
    init_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kale", "__init__.py")
    title = os.environ.get("PR_TITLE", "")

    try:
        version = find_release_version(title, read_package_version(init_file))
    except ReleaseTitleError as error:
        sys.stdout.write(f"::error::{error}\n")
        return 1

    if version is None:
        sys.stdout.write("::notice::Not a release pull request, skipping changelog generation.\n")
    else:
        sys.stdout.write(f"Release version: {version}\n")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version or ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
