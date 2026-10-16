$ErrorActionPreference = 'Stop'
python (Join-Path $PSScriptRoot 'scripts/run_markdown_domain.py') --with-tests @args
exit $LASTEXITCODE
