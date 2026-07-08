# reset_match_results.ps1

cd C:\Users\FuRyu\Desktop\tekken8_auto_tracker

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

mkdir logs -ErrorAction SilentlyContinue
mkdir archive -ErrorAction SilentlyContinue

if (Test-Path logs\match_results.csv) {
    Copy-Item logs\match_results.csv "archive\match_results_$stamp.csv"
}

@"
timestamp,result,score,margin,lose_score,none_score,win_score
"@ | Set-Content logs\match_results.csv -Encoding utf8

Write-Host "match_results.csv reset completed"
Write-Host "backup: archive\match_results_$stamp.csv"