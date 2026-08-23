# Builds the distribution tree in dist/.
#
# dist/ is the runtime root: fact-extractor.exe looks for settings.json and its
# data files next to itself.
#
# This script is the only thing that ever writes settings.json. The binary reads
# it and never modifies it. The values below come from hardware-finetune.md,
# which stays authoritative - if the two disagree, that file is right.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dist = Join-Path $root 'dist'

# The .gguf the Ollama model is created from. Tried next to the repo first, then
# the development path, so the same script works on either.
$modelName = 'fact-extractor'
$ggufCandidates = @(
    (Join-Path $root 'models\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf'),
    'D:\Modelos\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf'
)

# The second model exists so two can be scored on the same corpus with
# --coder. It is optional: without it the build still produces a working tool,
# and --coder reports that coder_model is empty.
$coderModelName = 'fact-extractor-coder'
$coderGgufCandidates = @(
    (Join-Path $root 'models\qwen2.5-coder-7b-instruct-q4_k_m.gguf'),
    'D:\Modelos\stefancosma\Qwen2.5-Coder-7B-Instruct-Q4_K_M-GGUF\qwen2.5-coder-7b-instruct-q4_k_m.gguf'
)

New-Item -ItemType Directory -Force -Path $dist, "$dist\corpus", "$dist\schemas" | Out-Null

Write-Host 'building fact-extractor.exe' -ForegroundColor Cyan
& go build -ldflags='-s -w' -o "$dist\fact-extractor.exe" $root
if ($LASTEXITCODE -ne 0) { throw 'go build failed for fact-extractor' }

# checkfacts validates a result.json independently of the tool that produced it.
Write-Host 'building checkfacts.exe' -ForegroundColor Cyan
& go build -ldflags='-s -w' -o "$dist\checkfacts.exe" "$root\cmd\checkfacts"
if ($LASTEXITCODE -ne 0) { throw 'go build failed for checkfacts' }

Copy-Item "$root\system-instruction.md" "$dist\" -Force
Copy-Item "$root\schemas\facts.json"    "$dist\schemas\" -Force
Copy-Item "$root\corpus\*"              "$dist\corpus\" -Force
Copy-Item "$root\hardware-finetune.md"  "$dist\" -Force

if (-not (Test-Path "$dist\prompt.md")) {
    Copy-Item "$root\corpus\01-news.md" "$dist\prompt.md"
}

# ---------------------------------------------------------------------------
# The service environment
#
# Computed here, before anything talks to Ollama, because `ollama create` puts
# blobs wherever the *server* points - so the service this script starts must
# run with the same OLLAMA_MODELS that goes into settings.json. Reading the
# value from the registry and not applying it is how models end up created in
# one store and looked for in another.
# ---------------------------------------------------------------------------

$modelsDir = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
if (-not $modelsDir) { $modelsDir = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'Machine') }

$serviceEnv = [ordered]@{
    OLLAMA_FLASH_ATTENTION = '1'
    OLLAMA_KV_CACHE_TYPE   = 'q8_0'
    OLLAMA_NUM_PARALLEL    = '1'
}
if ($modelsDir) { $serviceEnv['OLLAMA_MODELS'] = $modelsDir }

# Apply to this process so anything it launches inherits it. setx only affects
# new processes, so a shell opened before OLLAMA_MODELS was set does not have
# it even though the registry does.
foreach ($name in $serviceEnv.Keys) {
    Set-Item -Path "Env:$name" -Value $serviceEnv[$name]
}

# ---------------------------------------------------------------------------
# The Ollama models
#
# The Modelfile carries FROM and nothing else. Every PARAMETER line here would
# become a default that request options then override, which is two sources of
# truth for one value - exactly the drift settings.json exists to prevent.
#
# `ollama create` needs a service already listening. We start one if none is,
# and stop it again at the end - only if we were the ones who started it. That
# is the same adopt-or-manage rule the CLI follows, so the build behaves like
# the tool it builds.
# ---------------------------------------------------------------------------

function Test-OllamaUp {
    try {
        Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 -UseBasicParsing | Out-Null
        return $true
    } catch {
        return $false
    }
}

function New-OllamaModel([string]$name, [string]$gguf, [string]$modelfilePath) {
    "FROM `"$gguf`"" | Set-Content -Path $modelfilePath -Encoding utf8
    Write-Host "creating Ollama model '$name' from $(Split-Path $gguf -Leaf)" -ForegroundColor Cyan
    & ollama create $name -f $modelfilePath
    if ($LASTEXITCODE -ne 0) { throw "ollama create failed for $name" }
}

$gguf      = $ggufCandidates      | Where-Object { Test-Path $_ } | Select-Object -First 1
$coderGguf = $coderGgufCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $gguf) {
    Write-Host 'No .gguf found for the main model - skipping its creation. Looked in:' -ForegroundColor Yellow
    $ggufCandidates | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}
if (-not $coderGguf) {
    Write-Host 'No coder .gguf found - skipping it. --coder will report coder_model is empty.' -ForegroundColor Yellow
    $coderGgufCandidates | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}

$ollamaOnPath = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
if (($gguf -or $coderGguf) -and -not $ollamaOnPath) {
    Write-Host 'ollama is not on PATH - skipping model creation.' -ForegroundColor Yellow
    Write-Host "  Run later:  ollama create $modelName -f $dist\Modelfile" -ForegroundColor Yellow
}

$startedOllama = $null
if (($gguf -or $coderGguf) -and $ollamaOnPath) {
    if (Test-OllamaUp) {
        Write-Host 'using the Ollama service already running (it will be left running).' -ForegroundColor DarkGray
    } else {
        Write-Host 'starting ollama serve so models can be created' -ForegroundColor Cyan
        $startedOllama = Start-Process -FilePath 'ollama' -ArgumentList 'serve' -PassThru -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(60)
        while (-not (Test-OllamaUp)) {
            if ((Get-Date) -gt $deadline) {
                Stop-Process -Id $startedOllama.Id -Force -ErrorAction SilentlyContinue
                throw 'ollama serve did not answer within 60 seconds'
            }
            Start-Sleep -Milliseconds 500
        }
    }

    try {
        if ($gguf)      { New-OllamaModel $modelName      $gguf      (Join-Path $dist 'Modelfile') }
        if ($coderGguf) { New-OllamaModel $coderModelName $coderGguf (Join-Path $dist 'Modelfile.coder') }
    } finally {
        # Stop only what this script started; leave a service the user was
        # already running alone. Killing the tree matters - the runner
        # subprocesses hold the VRAM.
        if ($startedOllama) {
            Write-Host 'stopping the ollama service this build started' -ForegroundColor DarkGray
            & taskkill /T /F /PID $startedOllama.Id 2>&1 | Out-Null
        }
    }
}

if (-not $gguf) { $gguf = $ggufCandidates[0] }

# ---------------------------------------------------------------------------
# settings.json - generated here, read-only at run time
# ---------------------------------------------------------------------------

# The service env is what makes the VRAM budget hold (hardware-finetune.md
# section 4). With manage = true the CLI starts `ollama serve` itself when none
# is running and applies these - settings.json owns the whole tuning story.
# $serviceEnv was computed above, before the models were created, so the store
# the build wrote to and the store the CLI reads from cannot disagree.
$settings = [ordered]@{
    ollama = [ordered]@{
        host        = 'http://127.0.0.1:11434'
        model       = $modelName
        # Empty when no coder .gguf was found, which is what makes --coder fail
        # with a clear message rather than a missing-model error from Ollama.
        coder_model = $(if ($coderGguf) { $coderModelName } else { '' })
        keep_alive  = 0         # unload as soon as the run ends; hold no VRAM
    }
    service = [ordered]@{
        manage                  = $true
        command                 = 'ollama'
        startup_timeout_seconds = 60
        env                     = $serviceEnv
    }
    source_model = $gguf        # provenance, not a runtime path
    chunk_tokens = 3000
    options      = [ordered]@{
        num_ctx        = 16384  # always explicit: Ollama silently shrinks an automatic context
        num_gpu        = 99
        num_batch      = 512
        temperature    = 0.0
        top_k          = 1
        top_p          = 1.0
        repeat_penalty = 1.0
        seed           = 42
    }
}
$settings | ConvertTo-Json -Depth 5 | Set-Content -Path "$dist\settings.json" -Encoding utf8
Write-Host 'wrote settings.json' -ForegroundColor Cyan

# ---------------------------------------------------------------------------

Write-Host ''
Write-Host "dist ready: $dist" -ForegroundColor Green
Get-ChildItem $dist -Name | Sort-Object | ForEach-Object { Write-Host "  $_" }

# These three are service-level and cannot live in settings.json. The VRAM
# budget in hardware-finetune.md depends on them, so a missing one is worth
# saying out loud rather than discovering as unexplained slowness.
Write-Host ''
$required = @{
    OLLAMA_FLASH_ATTENTION = '1'
    OLLAMA_KV_CACHE_TYPE   = 'q8_0'
    OLLAMA_NUM_PARALLEL    = '1'
}
$missing = @()
foreach ($name in $required.Keys) {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, 'Machine') }
    if ($value -ne $required[$name]) { $missing += $name }
}
if ($missing.Count -gt 0) {
    Write-Host 'Note: the machine-wide Ollama environment differs from settings.json:' -ForegroundColor Yellow
    foreach ($name in $missing) {
        Write-Host ("  setx {0} {1}" -f $name, $required[$name]) -ForegroundColor Yellow
    }
    Write-Host '  Harmless when the CLI starts the service (it applies these itself), but an' -ForegroundColor Yellow
    Write-Host '  Ollama you run independently will not have them. hardware-finetune.md section 4.' -ForegroundColor Yellow
} else {
    Write-Host 'Machine-wide Ollama environment matches settings.json.' -ForegroundColor Green
}
