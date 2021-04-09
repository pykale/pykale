"""
Ref1: https://stackoverflow.com/questions/33767627/python-write-unittest-for-console-print
Ref2: https://docs.pytest.org/en/latest/capture.html#accessing-captured-output-from-a-test-function
"""

from kale.utils.print import pprint, pprint_without_newline, tprint


def test_tprint(capsys):
    tprint("hello")
    captured = capsys.readouterr()
    assert captured.out == "\rhello"


def test_pprint(capsys):
    pprint("hello")  # noqa: T003
    captured = capsys.readouterr()
    assert captured.out == "\rhello\n"


def test_pprint_without_newline(capsys):
    pprint_without_newline("hello")
    captured = capsys.readouterr()
    assert captured.out == "\rhello "
