"""Inspect Windows launch scripts without installing or starting any service."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPTS = [SCRIPT_DIR / name for name in (
    "docker-common.ps1", "setup-docker.ps1", "start-docker.ps1", "stop-docker.ps1")]
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
