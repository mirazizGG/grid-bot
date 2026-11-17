Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:GridBotRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:GridBotProject = 'grid-bot-home'
$script:GridBotCompose = Join-Path $script:GridBotRoot 'compose.yaml'
$script:GridBotState = Join-Path $script:GridBotRoot '.docker-state'
$script:GridBotBridge = Join-Path $script:GridBotState 'bridge'
$script:GridBotHostPython = Join-Path $script:GridBotRoot '.venv-host\Scripts\python.exe'
$script:GridBotHostPythonw = Join-Path $script:GridBotRoot '.venv-host\Scripts\pythonw.exe'
$script:GridBotBridgeScript = Join-Path $script:GridBotRoot 'backend\host_bridge.py'
$script:GridBotPort = 8811
if ($env:GRID_BOT_HTTP_PORT) {
    if ($env:GRID_BOT_HTTP_PORT -notmatch '\A[0-9]{1,5}\z' -or
        [int]$env:GRID_BOT_HTTP_PORT -lt 1 -or [int]$env:GRID_BOT_HTTP_PORT -gt 65535) {
        throw 'GRID_BOT_HTTP_PORT 1 dan 65535 gacha butun son bo''lishi kerak.'
    }
    $script:GridBotPort = [int]$env:GRID_BOT_HTTP_PORT
}

function Assert-GridWindows {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'Bu skript Windows uchun. Docker Desktop va Windows MT5 terminali kerak.'
    }
}

function Invoke-GridChecked {
    param([string]$FilePath, [string[]]$Arguments, [string]$FailureMessage)
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) { throw $FailureMessage }
    # Windows PowerShell can treat native stderr as an ErrorRecord. The native
    # exit code decides success; ordinary pip/build warnings must not abort setup.
    $ErrorActionPreference = 'Continue'
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

function Get-GridPythonInfo {
    param([string]$FilePath, [string[]]$Prefix = @())
    try {
        $probe = "import json,struct,sys; print(json.dumps({'path':sys.executable,'major':sys.version_info.major,'minor':sys.version_info.minor,'bits':struct.calcsize('P')*8}))"
        $output = & $FilePath @Prefix '-c' $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $info = ($output -join "`n") | ConvertFrom-Json
        if ($info.major -eq 3 -and $info.minor -eq 11 -and $info.bits -eq 64) { return $info }
    } catch { }
    return $null
}

function Find-GridPython {
    param([string]$RequestedPath = '')
    if ($RequestedPath) {
        $info = Get-GridPythonInfo -FilePath $RequestedPath
        if ($null -eq $info) { throw 'Ko''rsatilgan Python 3.11 x64 emas yoki ishga tushmadi.' }
        return $info.path
    }
    $launcher = Get-Command 'py.exe' -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) {
        $info = Get-GridPythonInfo -FilePath $launcher.Path -Prefix @('-3.11')
        if ($null -ne $info) { return $info.path }
    }
    $candidates = @()
    $command = Get-Command 'python.exe' -CommandType Application -ErrorAction SilentlyContinue
    if ($command -and $command.Path -notmatch '\\Microsoft\\WindowsApps\\') { $candidates += $command.Path }
    if ($env:LOCALAPPDATA) { $candidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe' }
    if ($env:ProgramFiles) { $candidates += Join-Path $env:ProgramFiles 'Python311\python.exe' }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $info = Get-GridPythonInfo -FilePath $candidate
            if ($null -ne $info) { return $info.path }
        }
    }
    throw ('Python 3.11 x64 topilmadi. Uni python.org orqali yoki o''zingiz quyidagi buyruq bilan o''rnating: ' +
        'winget install --exact --id Python.Python.3.11 --architecture x64 . So''ng yangi PowerShell oynasida setup-docker.ps1 ni qayta ishlating.')
}

function Assert-GridDocker {
    $ErrorActionPreference = 'Continue'
    $command = Get-Command 'docker.exe' -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) {
        throw ('Docker Desktop topilmadi. O''zingiz o''rnating: winget install --exact --id Docker.DockerDesktop . ' +
            'Installer shartlarini ko''rib chiqing, Docker Desktop ni ishga tushiring va Linux containers rejimini tanlang. ' +
            'Skript litsenziya shartlarini tasdiqlamaydi va Windows ni qayta boshlamaydi.')
    }
    $script:GridBotDocker = $command.Path
    $server = & $script:GridBotDocker 'info' '--format' '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Docker daemon javob bermadi. Docker Desktop ni oching, engine tayyor bo''lishini kuting va qayta urinib ko''ring.' }
    if (($server -join '').Trim() -ne 'linux') { throw 'Docker Linux containers rejimida bo''lishi kerak. Docker Desktop sozlamalarini tekshiring.' }
    $version = & $script:GridBotDocker 'compose' 'version' '--short' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $version) { throw 'Docker Compose V2 mavjud emas. Docker Desktop ni yangilang.' }
    if (-not (Test-Path -LiteralPath $script:GridBotCompose -PathType Leaf)) { throw 'compose.yaml topilmadi. Loyihaning to''liq arxivini bitta papkaga oching.' }
}

function Get-GridComposeArguments {
    return @('compose', '--project-name', $script:GridBotProject, '--project-directory', $script:GridBotRoot, '--file', $script:GridBotCompose)
}

function Get-GridCanonicalPath {
    param([string]$Path, [string]$PythonPath = $script:GridBotHostPython)
    $ErrorActionPreference = 'Continue'
    # realpath accepts an existing Desktop junction without trusting new links in state.
    $probe = "import json,os,sys; print(json.dumps({'path':os.path.normcase(os.path.realpath(sys.argv[1]))}))"
    $value = & $PythonPath '-c' $probe $Path 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $value) { throw 'Loyiha yo''lini tekshirib bo''lmadi.' }
    return (($value -join '') | ConvertFrom-Json).path
}

function Assert-GridProjectOwnership {
    param([string]$PythonPath = $script:GridBotHostPython)
    $ErrorActionPreference = 'Continue'
    $containers = @(& $script:GridBotDocker 'ps' '--all' '--filter' "label=com.docker.compose.project=$script:GridBotProject" '--quiet' 2>$null)
    if ($LASTEXITCODE -ne 0) { throw 'Mavjud Docker loyihalarini tekshirib bo''lmadi.' }
    $containers = @($containers | Where-Object { $_ })
    if ($containers.Count -eq 0) { return }
    $labels = @(& $script:GridBotDocker 'inspect' '--format' '{{json .Config.Labels}}' @containers 2>$null)
    if ($LASTEXITCODE -ne 0) { throw 'Docker loyihasining papkasini tekshirib bo''lmadi.' }
    $expected = Get-GridCanonicalPath -Path $script:GridBotRoot -PythonPath $PythonPath
    foreach ($value in $labels) {
        $properties = ($value | ConvertFrom-Json).PSObject.Properties
        $property = $properties['com.docker.compose.project.working_dir']
        $directory = if ($null -ne $property) { $property.Value } else { '' }
        if (-not $directory -or (Get-GridCanonicalPath -Path $directory -PythonPath $PythonPath) -ne $expected) {
            throw 'grid-bot-home nomli boshqa o''rnatma mavjud. Uni o''z papkasidagi stop-docker.ps1 bilan to''xtating. Boshqa o''rnatma avtomatik o''zgartirilmaydi.'
        }
    }
}

function Test-GridPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connected = $client.ConnectAsync('127.0.0.1', $script:GridBotPort)
        return ($connected.Wait(500) -and $client.Connected)
    } catch { return $false }
    finally { $client.Dispose() }
}

function Assert-GridPortAvailable {
    $ErrorActionPreference = 'Continue'
    if (-not (Test-GridPort)) { return }
    $containers = @(& $script:GridBotDocker 'ps' '--filter' "label=com.docker.compose.project=$script:GridBotProject" '--filter' "publish=$script:GridBotPort" '--format' '{{.ID}}' 2>$null)
    if ($LASTEXITCODE -ne 0 -or @($containers | Where-Object { $_ }).Count -eq 0) {
        throw ("127.0.0.1:{0} porti boshqa dastur tomonidan band. Uni o'zingiz tekshiring yoki GRID_BOT_HTTP_PORT bilan bo'sh port tanlang. Skript mavjud serverni o'chirmaydi." -f $script:GridBotPort)
    }
}

function Assert-GridNoStateLink {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw '.docker-state ichidagi havola yoki junction ruxsat etilmaydi. Oddiy mahalliy papkadan foydalaning.'
        }
    }
}

function Initialize-GridState {
    foreach ($relative in @('', 'data', 'data\logs', 'models', 'backups', 'bridge', 'bridge\requests', 'bridge\responses')) {
        $path = if ($relative) { Join-Path $script:GridBotState $relative } else { $script:GridBotState }
        Assert-GridNoStateLink -Path $path
        [void][IO.Directory]::CreateDirectory($path)
    }
    $token = Join-Path $script:GridBotBridge 'token'
    Assert-GridNoStateLink -Path $token
    if (-not (Test-Path -LiteralPath $token)) {
        $bytes = New-Object byte[] 32
        $random = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $random.GetBytes($bytes) } finally { $random.Dispose() }
        $secret = [BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant()
        $stream = [IO.File]::Open($token, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $encoded = [Text.Encoding]::ASCII.GetBytes($secret)
            $stream.Write($encoded, 0, $encoded.Length)
            $stream.Flush($true)
        } finally { $stream.Dispose() }
    }
    if ([IO.File]::ReadAllText($token).Trim() -notmatch '\A[0-9a-fA-F]{64}\z') {
        throw 'Windows ko''prigi token fayli yaroqsiz. Mavjud token avtomatik almashtirilmaydi.'
    }
}

function Assert-GridHostEnvironment {
    foreach ($relative in @('.venv-host', '.venv-host\Scripts', '.venv-host\Scripts\python.exe', '.venv-host\Scripts\pythonw.exe')) {
        Assert-GridNoStateLink -Path (Join-Path $script:GridBotRoot $relative)
    }
    if (-not (Test-Path -LiteralPath $script:GridBotHostPython -PathType Leaf) -or
        -not (Test-Path -LiteralPath $script:GridBotHostPythonw -PathType Leaf)) {
        throw 'Windows yordamchi muhiti topilmadi. Avval scripts\setup-docker.ps1 ni ishga tushiring.'
    }
    if ($null -eq (Get-GridPythonInfo -FilePath $script:GridBotHostPython)) {
        throw '.venv-host muhiti Python 3.11 x64 bilan yaratilmagan. setup-docker.ps1 ko''rsatmalarini tekshiring.'
    }
    if (-not (Test-Path -LiteralPath $script:GridBotBridgeScript -PathType Leaf)) { throw 'backend\host_bridge.py topilmadi. Loyiha arxivi to''liq emas.' }
}

function Invoke-GridHostBridge {
    param([ValidateSet('check', 'stop')][string]$Action)
    if (-not (Test-Path -LiteralPath $script:GridBotHostPython -PathType Leaf) -or
        -not (Test-Path -LiteralPath $script:GridBotBridgeScript -PathType Leaf)) { return 1 }
    $ErrorActionPreference = 'Continue'
    $previous = [Environment]::GetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', $null, 'Process')
        & $script:GridBotHostPython $script:GridBotBridgeScript '--directory' $script:GridBotBridge "--$Action" *> $null
        return $LASTEXITCODE
    } finally {
        [Environment]::SetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', $previous, 'Process')
    }
}

function Test-GridBridge {
    return ((Invoke-GridHostBridge -Action 'check') -eq 0)
}

function Start-GridBridge {
    if ((Test-Path -LiteralPath (Join-Path $script:GridBotBridge 'bridge.pid.json')) -and (Test-GridBridge)) { return }
    # Start-Process joins its argument array on Windows; quote each absolute path.
    $arguments = @(('"' + $script:GridBotBridgeScript + '"'), '--directory', ('"' + $script:GridBotBridge + '"'))
    $previous = [Environment]::GetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', $null, 'Process')
        $process = Start-Process -FilePath $script:GridBotHostPythonw -ArgumentList $arguments -WorkingDirectory $script:GridBotRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $script:GridBotState 'data\logs\host-bridge.out.log') -RedirectStandardError (Join-Path $script:GridBotState 'data\logs\host-bridge.error.log')
    } finally {
        [Environment]::SetEnvironmentVariable('GRID_BOT_BRIDGE_DIR', $previous, 'Process')
    }
    $deadline = [DateTime]::UtcNow.AddSeconds(35)
    do {
        if (Test-GridBridge) { return }
        $process.Refresh()
        if ($process.HasExited) { break }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'Windows MT5 yordamchisi tayyor bo''lmadi. .docker-state\data\logs\host-bridge.error.log faylini tekshiring. Mavjud jarayonlar majburan to''xtatilmaydi.'
}
