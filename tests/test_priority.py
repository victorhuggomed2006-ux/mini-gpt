from __future__ import annotations

import subprocess
import sys

import pytest

CODE = (
    "import ctypes\n"
    "from ctypes import wintypes\n"
    "from mini_gpt.train import lower_priority\n"
    "lower_priority()\n"
    "k = ctypes.WinDLL('kernel32')\n"
    "k.GetCurrentProcess.restype = wintypes.HANDLE\n"
    "k.GetPriorityClass.argtypes = [wintypes.HANDLE]\n"
    "k.GetPriorityClass.restype = wintypes.DWORD\n"
    "print(k.GetPriorityClass(k.GetCurrentProcess()))\n"
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
def test_lower_priority_really_sets_idle_priority() -> None:
    """The process must end up in the IDLE priority class (0x40), not just ask for it."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CODE], capture_output=True, text=True, check=True
    )
    assert int(result.stdout.strip()) == 0x40
