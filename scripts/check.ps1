param(
    [switch]$Full,
    [switch]$Sql
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

if ($Full) {
    python src/run_all.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Sql) {
    python src/07_sql_pipeline.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python -m unittest discover -s tests -v
exit $LASTEXITCODE
