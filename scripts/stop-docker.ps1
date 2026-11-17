[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'docker-common.ps1')

$failed = $false
try {
    Assert-GridWindows
    Assert-GridHostEnvironment
    Assert-GridDocker
    Assert-GridProjectOwnership
    $arguments = @(Get-GridComposeArguments) + @('down')
    Invoke-GridChecked -FilePath $script:GridBotDocker -Arguments $arguments -FailureMessage 'Docker loyihasini to''xtatib bo''lmadi.'
    Write-Host 'Grid-Bot Docker loyihasi to''xtatildi. .docker-state dagi ma''lumotlar saqlanadi.'
} catch {
    $failed = $true
    Write-Warning $_.Exception.Message
}
try {
    if ((Test-Path -LiteralPath $script:GridBotHostPython -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $script:GridBotBridge 'token') -PathType Leaf)) {
        if (Test-GridBridge) {
            if ((Invoke-GridHostBridge -Action 'stop') -ne 0) {
                throw 'Windows yordamchisi to''xtash so''roviga javob bermadi; jarayonlar majburan o''chirilmaydi.'
            }
            Write-Host 'Windows yordamchisiga to''xtash so''rovi yuborildi; joriy vazifalar tugagach yopiladi.'
        } elseif (Test-Path -LiteralPath (Join-Path $script:GridBotBridge 'bridge.pid.json')) {
            throw 'Windows yordamchisi javob bermadi. Uning jurnallarini tekshiring; jarayonlar majburan o''chirilmaydi.'
        } else {
            Write-Host 'Tayyor Windows yordamchisi topilmadi; to''xtatish amali kerak emas.'
        }
    }
} catch {
    $failed = $true
    Write-Warning $_.Exception.Message
}
if ($failed) { exit 1 }
