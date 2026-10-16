param([int]$MaxTokens = 24, [int]$Threads = 4)
$ErrorActionPreference = 'Stop'
python (Join-Path $PSScriptRoot 'scripts/run_part1.py') --max-tokens $MaxTokens --threads $Threads
exit $LASTEXITCODE
