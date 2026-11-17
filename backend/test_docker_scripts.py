"""Inspect Windows launch scripts without installing or starting any service."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPTS = [SCRIPT_DIR / name for name in (
    "docker-common.ps1", "setup-docker.ps1", "start-docker.ps1", "stop-docker.ps1", "update-docker.ps1")]
SHELLS = [path for name in ("powershell.exe", "pwsh.exe", "pwsh")
          if (path := shutil.which(name))]
SHELLS = list(dict.fromkeys(SHELLS))


def source(name):
    return (SCRIPT_DIR / name).read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("shell", SHELLS or [None])
def test_powershell_parses_and_launches_only_hidden_helper(shell):
    if shell is None:
        pytest.skip("PowerShell parser is unavailable on this platform")
    # ParseFile builds syntax trees only. None of the inspected scripts execute.
    command = r"""
        $ErrorActionPreference = 'Stop'
        $files = $env:GRID_BOT_TEST_SCRIPTS | ConvertFrom-Json
        $results = @()
        foreach ($file in $files) {
            $tokens = $null
            $errors = $null
            $tree = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors)
            $commands = @($tree.FindAll({ param($node)
                $node -is [System.Management.Automation.Language.CommandAst]
            }, $true) | ForEach-Object {
                @{name=$_.GetCommandName(); text=$_.Extent.Text}
            })
            $results += @{file=$file; errors=@($errors | ForEach-Object {$_.Message}); commands=$commands}
        }
        ConvertTo-Json -InputObject $results -Depth 6 -Compress
    """
    environment = {**os.environ, "GRID_BOT_TEST_SCRIPTS": json.dumps([str(path) for path in SCRIPTS])}
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout.lstrip("\ufeff"))
    assert len(parsed) == len(SCRIPTS)
    assert all(not item["errors"] for item in parsed), parsed
    commands = [command for item in parsed for command in item["commands"]]
    launches = [command for command in commands if command["name"] == "Start-Process"]
    assert len(launches) == 1
    assert re.search(r"-WindowStyle\s+Hidden\b", launches[0]["text"])
    assert "-FilePath $script:GridBotHostPythonw" in launches[0]["text"]
    assert "-ArgumentList $arguments" in launches[0]["text"]
    blocked = {"Stop-Process", "Restart-Computer", "Stop-Computer", "Set-ExecutionPolicy",
               "Invoke-Expression", "Remove-Item", "taskkill", "winget", "cmd.exe", "cmd"}
    assert not blocked.intersection(command["name"] for command in commands)


def test_shared_paths_credentials_and_guarded_bridge_contract():
    common = source("docker-common.ps1")
    assert "Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')" in common
    assert "os.path.realpath(sys.argv[1])" in common  # existing Desktop junctions are valid
    for directory in ("'data'", "'models'", "'backups'", "'bridge'", "'bridge\\requests'", "'bridge\\responses'"):
        assert directory in common
    assert "RandomNumberGenerator]::Create()" in common
    assert "New-Object byte[] 32" in common
    assert "[IO.FileMode]::CreateNew" in common
    assert "Join-Path $script:GridBotBridge 'token'" in common
    assert r"\A[0-9a-fA-F]{64}\z" in common
    assert "bridge.pid.json" in common
    assert "Invoke-GridHostBridge -Action 'check'" in common
    assert "GRID_BOT_BRIDGE_DIR', $null, 'Process'" in common
    assert "GRID_BOT_BRIDGE_DIR', $previous, 'Process'" in common
    assert """('"' + $script:GridBotBridgeScript + '"')""" in common
    assert """('"' + $script:GridBotBridge + '"')""" in common
    assert not re.search(r"(?im)^\s*Write-(?:Host|Output|Verbose|Debug).*\$(?:secret|token|encoded|bytes)\b", common)


def test_setup_only_prepares_project_environment_and_manual_prerequisite_guidance():
    setup = source("setup-docker.ps1")
    common = source("docker-common.ps1")
    assert "Assert-GridProjectOwnership -PythonPath $python" in setup
    assert setup.index("Assert-GridPortAvailable") < setup.index("'pip'")
    assert "backend\\requirements-host.txt" in setup
    assert "@('-m', 'venv', $environment)" in setup
    assert "$info.major -eq 3 -and $info.minor -eq 11 -and $info.bits -eq 64" in common
    assert "Python.Python.3.11 --architecture x64" in common
    assert "winget install --exact --id Docker.DockerDesktop" in common
    for forbidden in ("--accept-package-agreements", "--accept-source-agreements", "Set-ExecutionPolicy",
                      "Restart-Computer", "docker.sock", "--privileged"):
        assert forbidden not in common + setup


def test_start_rejects_foreign_listener_and_uses_configured_loopback_port():
    common = source("docker-common.ps1")
    start = source("start-docker.ps1")
    assert "$script:GridBotProject = 'grid-bot-home'" in common
    assert "GRID_BOT_HTTP_PORT" in common and "$script:GridBotPort = 8811" in common
    assert "[int]$env:GRID_BOT_HTTP_PORT -gt 65535" in common
    assert "ConnectAsync('127.0.0.1', $script:GridBotPort)" in common
    assert '"publish=$script:GridBotPort"' in common
    assert "com.docker.compose.project.working_dir" in common
    assert "--project-directory" in common and "--file" in common
    assert start.index("Assert-GridPortAvailable") < start.index("Start-GridBridge")
    assert "@('up', '--detach', '--build')" in start
    assert '"http://127.0.0.1:{0}/health" -f $script:GridBotPort' in start
    assert '"Grid-Bot tayyor: http://127.0.0.1:{0}" -f $script:GridBotPort' in start


def test_stop_preserves_state_and_uses_signed_graceful_shutdown():
    stop = source("stop-docker.ps1")
    assert "Assert-GridProjectOwnership" in stop
    assert "@('down')" in stop
    assert "Invoke-GridHostBridge -Action 'stop'" in stop
    for forbidden in ("--volumes", "--rmi", "Stop-Process", "taskkill", "Remove-Item", "-Recurse"):
        assert forbidden not in stop


def test_windows_powershell_51_can_decode_all_script_sources():
    for path in SCRIPTS:
        raw = path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf") or raw.isascii(), f"{path.name}: non-ASCII source requires UTF-8 BOM"


UPDATE_REQUIRED = {
    "compose.yaml": "new compose", "Dockerfile": "new Dockerfile",
    "scripts/docker-common.ps1": "# common", "scripts/setup-docker.ps1": "# setup",
    "scripts/start-docker.ps1": "# start", "scripts/stop-docker.ps1": "# stop",
    "scripts/update-docker.ps1": "# update", "backend/mt5_clock.py": "# clock",
    "backend/requirements-host.txt": "# requirements", "backend/host_bridge.py": "# bridge",
    ".dockerignore": "new dotfile", "backend/uploads/.gitkeep": "",
}


def update_python():
    match = re.search(r"\$script:GridUpdatePython = @'\n(.*?)\n'@", source("update-docker.ps1"), re.S)
    assert match
    return match.group(1)


def make_update_zip(path, extra=None, missing=None):
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in {**UPDATE_REQUIRED, **(extra or {})}.items():
            if name != missing:
                entry = zipfile.ZipInfo("placeholder")
                # ZipInfo's Windows constructor silently normalizes backslashes;
                # preserve malicious raw ZIP member names for the reader test.
                entry.filename = "grid-bot-test/" + name
                archive.writestr(entry, value)


def run_update_files(mode, root, stage, archive=None):
    return subprocess.run([sys.executable, "-c", update_python(), mode, str(root), str(stage),
                           str(archive or "")], capture_output=True, text=True, timeout=30)


def test_update_preserves_runtime_and_dotfiles_with_source_backup(tmp_path):
    root, stage = tmp_path / "project", tmp_path / "staging"
    root.mkdir()
    stage.mkdir()
    state = [".docker-state/bridge/token", ".docker-state/data/runtime/settings.json",
             ".docker-state/models/numeric_model.joblib", ".venv-host/Scripts/python.exe",
             "backend/.env", "backend/analyses.db", "backend/server_log.txt",
             "backend/uploads/private.png", "backups/private.zip", "notes.txt"]
    for name in state:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private-user-state")
    (root / "compose.yaml").write_text("old compose", encoding="utf-8")
    (root / ".dockerignore").write_text("old dotfile", encoding="utf-8")
    archive = tmp_path / "source.zip"
    make_update_zip(archive)
    prepared = run_update_files("prepare", root, stage, archive)
    assert prepared.returncode == 0, prepared.stderr
    assert (root / "compose.yaml").read_text() == "old compose"
    installed = run_update_files("install", root, stage)
    assert installed.returncode == 0, installed.stderr
    assert (root / "compose.yaml").read_text() == "new compose"
    assert (root / ".dockerignore").read_text() == "new dotfile"
    assert (stage / "previous/compose.yaml").read_text() == "old compose"
    assert not (root / "backend/uploads/.gitkeep").exists()
    assert all((root / name).read_bytes() == b"private-user-state" for name in state)


@pytest.mark.parametrize("malicious", ["../escape.py", "/escape.py", "C:/escape.py",
    "backend/../../escape.py", "backend\\escape.py", "backend/CON.py", "backend/test.py:stream",
    "backend/.env", ".docker-state/bridge/token", ".venv-host/Scripts/python.exe",
    "backend/uploads/private.png", "backend/analyses.db", "backups/secret.zip"])
def test_update_rejects_unsafe_or_private_zip_before_source_mutation(tmp_path, malicious):
    root, stage = tmp_path / "project", tmp_path / "staging"
    root.mkdir()
    stage.mkdir()
    (root / "compose.yaml").write_text("old compose")
    archive = tmp_path / "source.zip"
    make_update_zip(archive, {malicious: "unsafe"})
    result = run_update_files("prepare", root, stage, archive)
    assert result.returncode != 0
    assert (root / "compose.yaml").read_text() == "old compose"
    assert not (stage / "manifest.json").exists()


@pytest.mark.parametrize("case", ["symlink", "extra_root", "missing_clock", "case_collision"])
def test_update_rejects_incomplete_or_ambiguous_archives(tmp_path, case):
    root, stage = tmp_path / "project", tmp_path / "staging"
    root.mkdir()
    stage.mkdir()
    archive = tmp_path / "source.zip"
    make_update_zip(archive, missing="backend/mt5_clock.py" if case == "missing_clock" else None)
    with zipfile.ZipFile(archive, "a") as bundle:
        if case == "symlink":
            entry = zipfile.ZipInfo("grid-bot-test/backend/link.py")
            entry.create_system = 3
            entry.external_attr = 0o120777 << 16
            bundle.writestr(entry, "../../outside")
        elif case == "extra_root":
            bundle.writestr("other/project.py", "bad")
        elif case == "case_collision":
            bundle.writestr("grid-bot-test/COMPOSE.YAML", "bad")
    result = run_update_files("prepare", root, stage, archive)
    assert result.returncode != 0
    assert not (stage / "manifest.json").exists()


def test_update_refuses_source_edits_since_preparation(tmp_path):
    root, stage = tmp_path / "project", tmp_path / "staging"
    root.mkdir()
    stage.mkdir()
    (root / "compose.yaml").write_text("old")
    archive = tmp_path / "source.zip"
    make_update_zip(archive)
    result = run_update_files("prepare", root, stage, archive)
    assert result.returncode == 0, result.stderr
    (root / "compose.yaml").write_text("new local edit")
    result = run_update_files("install", root, stage)
    assert result.returncode != 0
    assert (root / "compose.yaml").read_text() == "new local edit"
    assert not (root / "backend/mt5_clock.py").exists()


def test_update_rolls_back_written_source_if_copy_fails(tmp_path):
    root, stage = tmp_path / "project", tmp_path / "staging"
    root.mkdir()
    stage.mkdir()
    (root / "compose.yaml").write_text("old compose")
    archive = tmp_path / "source.zip"
    make_update_zip(archive)
    assert run_update_files("prepare", root, stage, archive).returncode == 0
    definitions = update_python().split("mode,root,stage=sys.argv[1:4]")[0]
    injection = r'''
root,stage=map(pathlib.Path,sys.argv[1:3])
original=atomic_copy
calls=0
def fail_third(source,destination):
    global calls
    calls+=1
    if calls==3:
        raise OSError('injected copy failure')
    return original(source,destination)
atomic_copy=fail_third
try:
    install(root,stage)
except OSError:
    pass
else:
    raise AssertionError('copy should fail')
'''
    result = subprocess.run([sys.executable, "-c", definitions + injection, str(root), str(stage)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (root / "compose.yaml").read_text() == "old compose"
    assert not (root / "Dockerfile").exists()
    assert not (root / "backend/mt5_clock.py").exists()


def test_update_rejects_redirected_destination(tmp_path):
    root, stage, outside = tmp_path / "project", tmp_path / "staging", tmp_path / "outside"
    root.mkdir()
    stage.mkdir()
    outside.mkdir()
    try:
        (root / "backend").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a test symlink requires unavailable Windows privilege")
    archive = tmp_path / "source.zip"
    make_update_zip(archive)
    result = run_update_files("prepare", root, stage, archive)
    assert result.returncode != 0
    assert not list(outside.iterdir())


@pytest.mark.parametrize("shell", SHELLS or [None])
def test_real_powershell_update_file_boundary_preserves_state_and_spaced_paths(tmp_path, shell):
    if shell is None:
        pytest.skip("PowerShell runtime is unavailable")
    root = tmp_path / "existing project with spaces"
    stage = tmp_path / "download staging with spaces"
    archive = tmp_path / "source archive with spaces.zip"
    root.mkdir()
    stage.mkdir()
    (root / "compose.yaml").write_text("original source", encoding="utf-8")
    private_paths = [".docker-state/bridge/token", ".docker-state/data/runtime/settings.json",
                     ".venv-host/Scripts/python.exe", "backend/.env", "backend/analyses.db",
                     "backend/uploads/private.png", "backups/private.zip"]
    for name in private_paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"private fixture bytes")
    make_update_zip(archive)
    # Exercise the actual PowerShell -> Invoke-GridChecked -> native Python
    # boundary. Extracting the Python string and running it from Python misses
    # Windows PowerShell 5.1's native-argument quote removal (the home failure).
    # No subprocess/service/network boundary is mocked or called here: only the
    # real local staging/copy function runs, against temporary fixture files.
    command = r"""
        . $env:GRID_UPDATE_SCRIPT
        $script:GridBotRoot = $env:GRID_UPDATE_ROOT
        $script:GridBotHostPython = $env:GRID_UPDATE_PYTHON
        Invoke-GridUpdateFiles -Mode 'prepare' -Stage $env:GRID_UPDATE_STAGE -Archive $env:GRID_UPDATE_ARCHIVE
        if ([IO.File]::ReadAllText((Join-Path $script:GridBotRoot 'compose.yaml')) -ne 'original source') { throw 'prepare changed source' }
        Invoke-GridUpdateFiles -Mode 'install' -Stage $env:GRID_UPDATE_STAGE
        Write-Output 'NATIVE_UPDATE_COMPLETE'
    """
    environment = {**os.environ, "GRID_UPDATE_SCRIPT": str(SCRIPT_DIR / "update-docker.ps1"),
                   "GRID_UPDATE_ROOT": str(root), "GRID_UPDATE_STAGE": str(stage),
                   "GRID_UPDATE_ARCHIVE": str(archive), "GRID_UPDATE_PYTHON": sys.executable}
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "NATIVE_UPDATE_COMPLETE" in result.stdout
    assert (root / "compose.yaml").read_text(encoding="utf-8") == "new compose"
    assert (root / ".dockerignore").read_text(encoding="utf-8") == "new dotfile"
    assert (stage / "previous/compose.yaml").read_text(encoding="utf-8") == "original source"
    assert all((root / name).read_bytes() == b"private fixture bytes" for name in private_paths)


def test_update_helpers_do_not_pass_program_source_through_native_c_argument():
    script = source("update-docker.ps1")
    files = script.split("function Invoke-GridUpdateFiles {", 1)[1].split("function Invoke-GridUpdateStop", 1)[0]
    wait = script.split("function Wait-GridUpdateBridgeStopped {", 1)[1].split("function Invoke-GridUpdateSetup", 1)[0]
    for function in (files, wait):
        assert not re.search(r"-Arguments\s+@\(\s*['\"]-c['\"]", function)


@pytest.mark.parametrize("shell", SHELLS or [None])
def test_real_powershell_bridge_wait_uses_only_temporary_unlocked_file(tmp_path, shell):
    if shell is None or os.name != "nt":
        pytest.skip("Windows PowerShell and msvcrt are required for this boundary test")
    stage = tmp_path / "wait staging with spaces"
    bridge = tmp_path / "isolated bridge with spaces"
    stage.mkdir()
    bridge.mkdir()
    lock = bridge / "bridge.lock"
    lock.write_bytes(b"0")
    token = bridge / "token"
    token.write_bytes(b"fixture token must remain unchanged")
    command = r"""
        . $env:GRID_UPDATE_SCRIPT
        $script:GridBotHostPython = $env:GRID_UPDATE_PYTHON
        $script:GridBotBridge = $env:GRID_UPDATE_BRIDGE
        Wait-GridUpdateBridgeStopped -Stage $env:GRID_UPDATE_STAGE
        Write-Output 'NATIVE_WAIT_COMPLETE'
    """
    environment = {**os.environ, "GRID_UPDATE_SCRIPT": str(SCRIPT_DIR / "update-docker.ps1"),
                   "GRID_UPDATE_PYTHON": sys.executable, "GRID_UPDATE_BRIDGE": str(bridge),
                   "GRID_UPDATE_STAGE": str(stage)}
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "NATIVE_WAIT_COMPLETE" in result.stdout
    assert lock.read_bytes() == b"0"
    assert token.read_bytes() == b"fixture token must remain unchanged"
    assert sorted(path.name for path in bridge.iterdir()) == ["bridge.lock", "token"]


@pytest.mark.parametrize("shell", SHELLS or [None])
@pytest.mark.parametrize("fail_stop", [False, True])
@pytest.mark.parametrize("clock", [False, True])
def test_updater_orders_prepare_stop_wait_copy_setup_without_live_actions(tmp_path, shell, fail_stop, clock):
    if shell is None:
        pytest.skip("PowerShell runtime is unavailable")
    # Dot sourcing defines functions only. All network/process/service boundaries
    # are replaced; the command exercises the real orchestration function.
    command = r"""
        . $env:GRID_UPDATE_SCRIPT
        $script:events = [System.Collections.Generic.List[string]]::new()
        function Assert-GridWindows {}
        function Assert-GridHostEnvironment {}
        function Assert-GridDocker {}
        function Assert-GridProjectOwnership {}
        function Assert-GridPortAvailable {}
        function Invoke-WebRequest { param($Uri,$OutFile,[switch]$UseBasicParsing,$TimeoutSec)
            $script:events.Add('download')
            if ($Uri -ne 'https://github.com/mirazizGG/grid-bot/archive/codex/docker-windows-dashboard.zip') { throw 'wrong source' }
        }
        function Invoke-GridUpdateFiles { param($Mode,$Stage,$Archive)
            $script:events.Add($Mode)
        }
        function Invoke-GridUpdateStop {
            $script:events.Add('stop')
            if ($env:GRID_FAIL_STOP -eq '1') { throw 'fake stop failure' }
        }
        function Wait-GridUpdateBridgeStopped { $script:events.Add('wait-helper') }
        function Invoke-GridUpdateSetup { $script:events.Add('setup') }
        function Invoke-GridChecked { param($FilePath,$Arguments,$FailureMessage)
            if ($env:GRID_TEST_CLOCK -ne '1') { throw 'must not diagnose without explicit clock parameters' }
            if ($FilePath -ne $script:GridBotHostPython) { throw 'wrong Python executable' }
            $expected = @((Join-Path $script:GridBotRoot 'backend\mt5_clock.py'),'--diagnose','--server','MetaQuotes-Demo','--offset-minutes','120','--apply-url',("http://127.0.0.1:{0}" -f $script:GridBotPort),'--terminal-path',$env:GRID_TEST_TERMINAL)
            if (($Arguments | ConvertTo-Json -Compress) -ne ($expected | ConvertTo-Json -Compress)) { throw 'wrong diagnostic arguments' }
            $script:events.Add('diagnose-apply')
        }
        function Invoke-RestMethod { param($Uri,$TimeoutSec)
            $script:events.Add('status')
            return @{ mt5 = @{status='ready'} }
        }
        $failure = $null
        try {
            if ($env:GRID_TEST_CLOCK -eq '1') {
                Invoke-GridDockerUpdate -SelectedRevision 'codex/docker-windows-dashboard' -Server 'MetaQuotes-Demo' -HasOffset $true -OffsetMinutes 120 -TerminalPath $env:GRID_TEST_TERMINAL | Out-Null
            } else {
                Invoke-GridDockerUpdate -SelectedRevision 'codex/docker-windows-dashboard' | Out-Null
            }
        }
        catch { $failure = $_.Exception.Message }
        Write-Output ('RESULT:' + (ConvertTo-Json -Compress -InputObject @{events=@($script:events.ToArray()); failure=$failure}))
    """
    terminal = tmp_path / "terminal64.exe"
    terminal.write_bytes(b"not executable; fixture only")
    environment = {**os.environ, "GRID_UPDATE_SCRIPT": str(SCRIPT_DIR / "update-docker.ps1"),
                   "GRID_TEST_CLOCK": "1" if clock else "0", "GRID_TEST_TERMINAL": str(terminal),
                   "GRID_FAIL_STOP": "1" if fail_stop else "0", "TEMP": str(tmp_path), "TMP": str(tmp_path)}
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    output = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT:")))
    if fail_stop:
        assert output == {"events": ["download", "prepare", "stop"], "failure": "fake stop failure"}
    else:
        events = ["download", "prepare", "stop", "wait-helper", "install", "setup"]
        if clock:
            events.append("diagnose-apply")
        assert output == {"events": events + ["status"], "failure": None}
