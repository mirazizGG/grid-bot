[CmdletBinding()]
param(
    [string]$Revision = 'codex/docker-windows-dashboard',
    [string]$Mt5Server = '',
    [ValidateRange(-840, 840)][int]$Mt5UtcOffsetMinutes,
    [string]$Mt5TerminalPath = ''
)

. (Join-Path $PSScriptRoot 'docker-common.ps1')

# Python is already installed in the owned host venv. Its ZIP/path checks work
# identically on Windows PowerShell 5.1 and PowerShell 7; no archive code runs.
$script:GridUpdatePython = @'
import hashlib,json,os,pathlib,re,shutil,stat,sys,tempfile,zipfile

def checked(root, relative):
    path = root.joinpath(*relative.split('/'))
    current = root
    for part in relative.split('/'):
        current = current / part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if current.is_symlink() or getattr(info,'st_file_attributes',0) & 1024:
                raise ValueError('Linked update destination is forbidden')
    if not path.resolve().is_relative_to(root) or path.resolve() != path:
        raise ValueError('Update destination escaped project')
    return path

def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()

def parts(name):
    if not name or '\\' in name or name.startswith('/'):
        raise ValueError('Invalid ZIP path')
    result = name.rstrip('/').split('/')
    for part in result:
        if (not part or part in ('.','..') or part[-1:] in (' ','.') or
            re.search(r'[\x00-\x1f<>:"|?*]',part) or
            re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part)):
            raise ValueError('Unsafe ZIP member')
    return result

def private(relative):
    path = relative.lower()
    pieces = path.split('/')
    if any(p in ('.git','.docker-state','.venv-host','.venv','venv','__pycache__') for p in pieces):
        return True
    if any(p.endswith('.env') or (p.startswith('.env.') and not p.endswith('.example')) for p in pieces):
        return True
    if pieces[0] in ('backups','dataset') or path == 'dataset_grid_bot.zip':
        return True
    if any(path.startswith(prefix) for prefix in ('backend/runtime/','backend/uploads/',
            'backend/_screens/','data_pipeline/output/','training/model_candidate/')):
        return True
    return path.endswith(('.db','.db-wal','.db-shm','.log')) or path in (
        'backend/server_log.txt','backend/server_error.log','data_pipeline/render_log.txt','training/zip_log.txt')

def prepare(archive, root, stage):
    source = stage / 'source'
    source.mkdir()
    previous = stage / 'previous'
    previous.mkdir()
    entries, seen, roots, total = [], set(), set(), 0
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist()) > 10000:
            raise ValueError('Too many ZIP entries')
        for entry in bundle.infolist():
            # ZipInfo normalizes Windows backslashes and truncates NULs in
            # filename; inspect the original member name before either change.
            components = parts(entry.orig_filename)
            roots.add(components[0])
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if kind not in (0,stat.S_IFREG,stat.S_IFDIR):
                raise ValueError('ZIP links/devices are forbidden')
            if entry.is_dir():
                continue
            if len(components) < 2:
                raise ValueError('ZIP requires one project root')
            relative = '/'.join(components[1:])
            if relative.lower() in seen:
                raise ValueError('Duplicate/case-colliding ZIP member')
            seen.add(relative.lower())
            if relative == 'backend/uploads/.gitkeep':
                continue  # tracked placeholder, never copy over runtime uploads
            if private(relative):
                raise ValueError('Private runtime data in source archive')
            total += entry.file_size
            if total > 512*1024*1024 or entry.flag_bits & 1:
                raise ValueError('Oversized/encrypted source archive')
            destination = checked(root,relative)
            if destination.exists() and not destination.is_file():
                raise ValueError('Source file conflicts with local directory')
            entries.append((entry,relative,destination))
        required = {'compose.yaml','Dockerfile','scripts/docker-common.ps1','scripts/setup-docker.ps1',
            'scripts/start-docker.ps1','scripts/stop-docker.ps1','scripts/update-docker.ps1',
            'backend/mt5_clock.py','backend/requirements-host.txt','backend/host_bridge.py'}
        if len(roots) != 1 or not required.issubset({relative for _,relative,_ in entries}):
            raise ValueError('Archive is not the complete tested Docker/clock revision')
        manifest = []
        for entry,relative,destination in entries:
            target = checked(source,relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(entry) as incoming, target.open('xb') as output:
                shutil.copyfileobj(incoming,output,1024*1024)
            old_hash = None
            if destination.exists():
                backup = checked(previous,relative)
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(destination,backup)
                old_hash = digest(backup)
            manifest.append({'path':relative,'sha256':digest(target),'old_sha256':old_hash})
    (stage/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

def atomic_copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.grid-update-',delete=False) as output:
            temporary = pathlib.Path(output.name)
            with source.open('rb') as incoming:
                shutil.copyfileobj(incoming,output,1024*1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary,destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def install(root,stage):
    manifest=json.loads((stage/'manifest.json').read_text(encoding='utf-8'))
    # Validate everything again before the first overwrite, including edits made
    # while the download or graceful stop was in progress.
    for entry in manifest:
        relative=entry['path']
        destination=checked(root,relative)
        if digest(checked(stage/'source',relative)) != entry['sha256']:
            raise ValueError('Staged source changed')
        old=digest(destination) if destination.is_file() else None
        if old != entry['old_sha256']:
            raise ValueError('Local source changed during update; nothing overwritten')
    written=[]
    try:
        for entry in manifest:
            relative=entry['path']
            atomic_copy(checked(stage/'source',relative),checked(root,relative))
            written.append(entry)
    except BaseException:
        for entry in reversed(written):
            destination=checked(root,entry['path'])
            if entry['old_sha256'] is None:
                destination.unlink(missing_ok=True)
            else:
                atomic_copy(checked(stage/'previous',entry['path']),destination)
        raise

mode,root,stage=sys.argv[1:4]
root=pathlib.Path(root).resolve()
stage=pathlib.Path(stage).resolve()
if mode=='prepare':
    prepare(sys.argv[4],root,stage)
elif mode=='install':
    install(root,stage)
else:
    raise ValueError('Unknown update operation')
'@

function Invoke-GridUpdateFiles {
    param([ValidateSet('prepare','install')][string]$Mode, [string]$Stage, [string]$Archive = '')
    Invoke-GridChecked -FilePath $script:GridBotHostPython -Arguments @('-c', $script:GridUpdatePython, $Mode, $script:GridBotRoot, $Stage, $Archive) -FailureMessage 'Update source validation/copy failed. Existing runtime data was preserved.'
}

function Invoke-GridUpdateStop {
    $shell = Join-Path $PSHOME 'powershell.exe'
    if (-not (Test-Path -LiteralPath $shell)) { $shell = Join-Path $PSHOME 'pwsh.exe' }
    Invoke-GridChecked -FilePath $shell -Arguments @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $script:GridBotRoot 'scripts\stop-docker.ps1')) -FailureMessage 'Owned application could not stop; source files were not replaced.'
}

function Wait-GridUpdateBridgeStopped {
    # stop-docker acknowledges the request before in-flight inference drains.
    # Wait for the actual lifetime lock, never terminate a PID from a file.
    $probe = @'
import pathlib,sys,time,msvcrt
root=pathlib.Path(sys.argv[1]).resolve()
deadline=time.monotonic()+630
while True:
    path=root/'bridge.lock'
    if path.is_symlink() or (path.exists() and getattr(path.lstat(),'st_file_attributes',0)&1024):
        raise RuntimeError('Linked bridge lock is forbidden')
    if not path.exists():
        if (root/'bridge.pid.json').exists():
            raise RuntimeError('Bridge ownership state is inconsistent')
        break
    with path.open('r+b') as lock:
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            pass
        else:
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            break
    if time.monotonic()>=deadline:
        raise RuntimeError('Bridge did not stop; no files replaced')
    time.sleep(0.25)
'@
    Invoke-GridChecked -FilePath $script:GridBotHostPython -Arguments @('-c',$probe,$script:GridBotBridge) -FailureMessage 'Windows helper is still active. Update stopped before replacing source.'
}

function Invoke-GridUpdateSetup {
    $shell = Join-Path $PSHOME 'powershell.exe'
    if (-not (Test-Path -LiteralPath $shell)) { $shell = Join-Path $PSHOME 'pwsh.exe' }
    Invoke-GridChecked -FilePath $shell -Arguments @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $script:GridBotRoot 'scripts\setup-docker.ps1'),'-PythonPath',$script:GridBotHostPython) -FailureMessage 'Updated source is saved, but application setup failed. Runtime data and source backup were preserved.'
}

function Invoke-GridDockerUpdate {
    param([string]$SelectedRevision, [string]$Server = '', [bool]$HasOffset = $false,
          [int]$OffsetMinutes = 0, [string]$TerminalPath = '')
    if ($SelectedRevision -cne 'codex/docker-windows-dashboard' -and $SelectedRevision -notmatch '\A[0-9a-fA-F]{40}\z') {
        throw 'Revision must be the approved branch or a full 40-character commit.'
    }
    if (($Server -ne '') -ne $HasOffset -or ($TerminalPath -ne '' -and -not $HasOffset)) {
        throw 'MT5 diagnosis requires both -Mt5Server and -Mt5UtcOffsetMinutes.'
    }
    if ($HasOffset -and -not $TerminalPath) {
        throw 'MT5 diagnosis also requires -Mt5TerminalPath for the intended terminal executable.'
    }
    if ($HasOffset -and ($OffsetMinutes -lt -840 -or $OffsetMinutes -gt 840 -or $Server.Length -gt 128 -or $Server -match '[\x00-\x1f]')) {
        throw 'Invalid explicit MT5 clock configuration.'
    }
    if ($TerminalPath -and (-not [IO.Path]::IsPathRooted($TerminalPath) -or -not (Test-Path -LiteralPath $TerminalPath -PathType Leaf) -or [IO.Path]::GetFileName($TerminalPath) -notin @('terminal64.exe','terminal.exe'))) {
        throw 'MT5 terminal path must identify an existing absolute executable path.'
    }
    Assert-GridWindows
    Assert-GridHostEnvironment
    Assert-GridDocker
    Assert-GridProjectOwnership
    Assert-GridPortAvailable
    Assert-GridNoStateLink -Path $script:GridBotState
    Assert-GridNoStateLink -Path $script:GridBotBridge
    $stage = Join-Path ([IO.Path]::GetTempPath()) ('grid-bot-update-' + [Guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($stage)
    $archive = Join-Path $stage 'source.zip'
    $url = 'https://github.com/mirazizGG/grid-bot/archive/' + $SelectedRevision + '.zip'
    Write-Host 'Downloading and validating the approved Grid-Bot source before stopping the application...'
    Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing -TimeoutSec 180
    Invoke-GridUpdateFiles -Mode 'prepare' -Stage $stage -Archive $archive
    Write-Host ('Prepared source backup: ' + (Join-Path $stage 'previous'))
    Invoke-GridUpdateStop
    Wait-GridUpdateBridgeStopped
    Invoke-GridUpdateFiles -Mode 'install' -Stage $stage
    Invoke-GridUpdateSetup
    if ($HasOffset) {
        $arguments = @((Join-Path $script:GridBotRoot 'backend\mt5_clock.py'),'--diagnose','--server',$Server,'--offset-minutes',([string]$OffsetMinutes),'--apply-url',("http://127.0.0.1:{0}" -f $script:GridBotPort))
        if ($TerminalPath) { $arguments += @('--terminal-path',$TerminalPath) }
        Invoke-GridChecked -FilePath $script:GridBotHostPython -Arguments $arguments -FailureMessage 'Application updated, but MT5 clock verification/application failed. No unverified profile was applied; inspect the diagnostic above.'
    }
    $services = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/services" -f $script:GridBotPort) -TimeoutSec 30
    $services.mt5 | ConvertTo-Json -Depth 6
    if ($HasOffset -and $services.mt5.status -ne 'ready') {
        throw 'MT5 is not ready after verification. Application and saved data remain available; see the MT5 result above.'
    }
    Write-Host ("Grid-Bot updated: http://127.0.0.1:{0}" -f $script:GridBotPort)
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        Invoke-GridDockerUpdate -SelectedRevision $Revision -Server $Mt5Server -HasOffset $PSBoundParameters.ContainsKey('Mt5UtcOffsetMinutes') -OffsetMinutes $Mt5UtcOffsetMinutes -TerminalPath $Mt5TerminalPath
    } catch {
        Write-Error -Message $_.Exception.Message -ErrorAction Continue
        exit 1
    }
}
