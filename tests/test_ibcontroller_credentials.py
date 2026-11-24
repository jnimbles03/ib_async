"""Account and FIX passwords must not be visible to other local users.

``IBC.startAsync`` is the only place this package builds the IBC command.
``IBC.start`` and ``Watchdog.runAsync`` both call it and do not assemble a
separate command line. These tests drive those entry points, then read the
child command line the way another account on the host would (``ps`` and
``/proc/<pid>/cmdline``).

The assertions fail while either secret is still placed on that command line.
"""

import asyncio
import os
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from ib_async.ib import IB
from ib_async.ibcontroller import IBC, Watchdog

# Obvious placeholders. These are not real credentials.
ACCOUNT_PASSWORD = "placeholder-account-password"
FIX_PASSWORD = "placeholder-fix-password"
ACCOUNT_USER = "placeholder-account-user"
FIX_USER = "placeholder-fix-user"

_LAUNCHER = """#!/bin/sh
trap 'exit 0' TERM INT
while :; do
    sleep 0.2
done
"""


def _install_launcher(ibc_path: Path, windows: bool) -> str:
    """Create the executable ``startAsync`` will exec, and return its basename."""
    ibc_path.mkdir(parents=True, exist_ok=True)
    if windows:
        # Same string ``startAsync`` builds on Windows:
        # f"{ibcPath}\\scripts\\StartIBC.bat"
        launcher = Path(str(ibc_path) + "\\scripts\\StartIBC.bat")
        basename = "StartIBC.bat"
    else:
        scripts = ibc_path / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        launcher = scripts / "ibcstart.sh"
        basename = "ibcstart.sh"
    launcher.write_text(_LAUNCHER)
    launcher.chmod(0o755)
    return basename


def _other_user_process_arguments(pid: int) -> str:
    """Return the command line of ``pid`` as seen by a different local user.

    ``/proc/<pid>/cmdline`` is the kernel file ``ps`` reads. This requires it
    to be world-readable, then reads it as ``nobody`` so the test is looking
    at another account's view rather than the owner's.
    """
    cmdline_path = f"/proc/{pid}/cmdline"
    mode = os.stat(cmdline_path).st_mode
    if mode & stat.S_IROTH == 0:
        raise AssertionError(
            f"{cmdline_path} is not readable by other users (mode {oct(mode)})"
        )
    cmdline = subprocess.run(
        ["sudo", "-n", "-u", "nobody", "cat", cmdline_path],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\0", b" ")
    listing = subprocess.run(
        [
            "sudo",
            "-n",
            "-u",
            "nobody",
            "ps",
            "-ww",
            "-p",
            str(pid),
            "-o",
            "args=",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return (cmdline + b"\n" + listing).decode("utf-8", "surrogateescape")


def _assert_secrets_hidden(
    visible: str, secrets: list[str], launcher_name: str
) -> None:
    assert launcher_name in visible
    for secret in secrets:
        assert secret not in visible


async def _stop(ibc: IBC) -> None:
    monitor = ibc._monitor
    proc = ibc._proc
    if monitor is not None:
        monitor.cancel()
    if proc is not None and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    if monitor is not None:
        with suppress(asyncio.CancelledError):
            await monitor
    ibc._proc = None
    ibc._monitor = None


def _make_ibc(
    ibc_path: Path,
    *,
    windows: bool,
    account_password: str,
    fix_password: str,
) -> IBC:
    platform_name = "win32" if windows else sys.platform
    with patch("ib_async.ibcontroller.sys.platform", platform_name):
        return IBC(
            1019,
            gateway=True,
            tradingMode="paper",
            ibcPath=str(ibc_path),
            userid=ACCOUNT_USER,
            password=account_password,
            fixuserid=FIX_USER,
            fixpassword=fix_password,
        )


async def _launch_and_read(
    ibc_path: Path,
    *,
    windows: bool,
    account_password: str,
    fix_password: str,
) -> tuple[str, str, list[str]]:
    launcher_name = _install_launcher(ibc_path, windows)
    ibc = _make_ibc(
        ibc_path,
        windows=windows,
        account_password=account_password,
        fix_password=fix_password,
    )
    await ibc.startAsync()
    try:
        proc = ibc._proc
        assert proc is not None and proc.returncode is None
        visible = _other_user_process_arguments(proc.pid)
    finally:
        await _stop(ibc)
    secrets = [secret for secret in (account_password, fix_password) if secret]
    return visible, launcher_name, secrets


@pytest.mark.parametrize("windows", [False, True], ids=["unix", "windows"])
@pytest.mark.parametrize(
    "account_password,fix_password",
    [
        (ACCOUNT_PASSWORD, ""),
        ("", FIX_PASSWORD),
        (ACCOUNT_PASSWORD, FIX_PASSWORD),
    ],
    ids=["account-password", "fix-password", "both-passwords"],
)
def test_start_async_does_not_publish_passwords_in_process_arguments(
    tmp_path: Path,
    windows: bool,
    account_password: str,
    fix_password: str,
) -> None:
    visible, launcher_name, secrets = asyncio.run(
        _launch_and_read(
            tmp_path / "ibc",
            windows=windows,
            account_password=account_password,
            fix_password=fix_password,
        )
    )
    _assert_secrets_hidden(visible, secrets, launcher_name)


def test_start_does_not_publish_passwords_in_process_arguments(tmp_path: Path) -> None:
    ibc_path = tmp_path / "ibc"
    launcher_name = _install_launcher(ibc_path, windows=False)
    ibc = _make_ibc(
        ibc_path,
        windows=False,
        account_password=ACCOUNT_PASSWORD,
        fix_password=FIX_PASSWORD,
    )
    ibc.start()
    try:
        proc = ibc._proc
        assert proc is not None and proc.returncode is None
        visible = _other_user_process_arguments(proc.pid)
    finally:
        ibc.terminate()
    _assert_secrets_hidden(visible, [ACCOUNT_PASSWORD, FIX_PASSWORD], launcher_name)


def test_watchdog_does_not_publish_passwords_in_process_arguments(
    tmp_path: Path,
) -> None:
    asyncio.run(_watchdog_launch(tmp_path / "ibc"))


async def _watchdog_launch(ibc_path: Path) -> None:
    launcher_name = _install_launcher(ibc_path, windows=False)
    ibc = _make_ibc(
        ibc_path,
        windows=False,
        account_password=ACCOUNT_PASSWORD,
        fix_password=FIX_PASSWORD,
    )
    watchdog = Watchdog(
        ibc,
        IB(),
        host="127.0.0.1",
        port=1,
        appStartupTime=30,
        connectTimeout=0.2,
        retryDelay=0.1,
    )
    runner = watchdog.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        while ibc._proc is None:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("Watchdog did not start the gateway process")
            await asyncio.sleep(0.05)
        proc = ibc._proc
        assert proc.returncode is None
        visible = _other_user_process_arguments(proc.pid)
        _assert_secrets_hidden(visible, [ACCOUNT_PASSWORD, FIX_PASSWORD], launcher_name)
    finally:
        watchdog.stop()
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
        await _stop(ibc)
