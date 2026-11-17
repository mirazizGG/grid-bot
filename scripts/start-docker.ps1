[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'docker-common.ps1')

try {
    Assert-GridWindows
    Assert-GridHostEnvironment
    Assert-GridDocker
    Assert-GridProjectOwnership
    Assert-GridPortAvailable
    Initialize-GridState
    Start-GridBridge
    $arguments = @(Get-GridComposeArguments) + @('up', '--detach', '--build')
    Write-Host 'Docker konteyneri yig''ilmoqda va ishga tushirilmoqda...'
    Invoke-GridChecked -FilePath $script:GridBotDocker -Arguments $arguments -FailureMessage 'Docker konteynerini ishga tushirib bo''lmadi. Docker Desktop va build xabarlarini tekshiring.'
    $ready = $false
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    do {
        try {
            $response = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/health" -f $script:GridBotPort) -UseBasicParsing -TimeoutSec 3
            $health = $response.Content | ConvertFrom-Json
            if ($response.StatusCode -eq 200 -and $health.status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)
    if (-not $ready) { throw 'Konteyner ishga tushdi, lekin server hali tayyor emas. docker compose --project-name grid-bot-home logs buyrug''i bilan jurnalni tekshiring.' }
    Write-Host ''
    Write-Host ("Grid-Bot tayyor: http://127.0.0.1:{0}" -f $script:GridBotPort)
    Write-Host 'Gemini kalitini Boshqaruv sahifasidagi Sozlamalarda kiriting. MT5 terminali Windows da ochiq va hisobga ulangan bo''lsin.'
} catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
