$ErrorActionPreference = 'Stop'
python (Join-Path $PSScriptRoot 'setup_cpu.py') @args
exit $LASTEXITCODE
