[CmdletBinding()]
param([string]$PythonPath = '')

. (Join-Path $PSScriptRoot 'docker-common.ps1')

try {
    Assert-GridWindows
    Assert-GridDocker
    $python = Find-GridPython -RequestedPath $PythonPath
    Assert-GridProjectOwnership -PythonPath $python
    Assert-GridPortAvailable
    $requirements = Join-Path $script:GridBotRoot 'backend\requirements-host.txt'
    if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) { throw 'backend\requirements-host.txt topilmadi. Loyiha arxivini to''liq oching.' }
    $environment = Join-Path $script:GridBotRoot '.venv-host'
    Assert-GridNoStateLink -Path $environment
    if (-not (Test-Path -LiteralPath $environment)) {
        Write-Host 'Windows MT5 yordamchisi uchun Python muhiti tayyorlanmoqda...'
        Invoke-GridChecked -FilePath $python -Arguments @('-m', 'venv', $environment) -FailureMessage 'Python muhitini yaratib bo''lmadi.'
    }
    Assert-GridHostEnvironment
    Write-Host 'Windows yordamchisi kutubxonalari o''rnatilmoqda...'
    Invoke-GridChecked -FilePath $script:GridBotHostPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirements) -FailureMessage 'Windows kutubxonalarini o''rnatib bo''lmadi. Internet va Python 3.11 x64 ni tekshiring.'
    Initialize-GridState
    & (Join-Path $PSScriptRoot 'start-docker.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Docker ishga tushirish bosqichi tugamadi. Yuqoridagi xabarni tekshiring.' }
} catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
