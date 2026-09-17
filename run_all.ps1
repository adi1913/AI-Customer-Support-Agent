$ErrorActionPreference = "Stop"

Write-Host "==> Training intent classifier..."
$env:PYTHONPATH = "src"
python src/train_classifier.py

Write-Host ""
Write-Host "==> Running evaluation..."
python eval/run_eval.py

Write-Host ""
Write-Host "==> Running end-to-end agent..."
$env:PYTHONPATH = "src"
python src/agent.py

Write-Host ""
Write-Host "Done. See report/REPORT.md for the full write-up."