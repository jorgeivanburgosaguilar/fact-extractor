# fact-extractor

**A toy project asking one question: is a local LLM better than I am at pulling the facts
out of a piece of text?**

Toy in scope, not in method. To answer that honestly you need a fair contest, so the
project is really three things: a CLI that makes a 7B model extract facts and *proves* every
citation against the source, a corpus of documents I read and extracted by hand, and a
benchmark that scores the model against my answers.

```powershell
fact-extractor.exe              # prompt.md -> result.json
fact-extractor.exe --benchmark  # score the model against my hand-extracted corpus
```

Model, context and sampling come from `settings.json`, which the build generates. There are
no other flags.

## The scoreboard

Each corpus case is a document I extracted by hand; the gold list is what *I* found. The
benchmark reports how much of my list the model recovered.

| Document | My facts | Model found | |
|---|---:|---:|---|
| `01-news` — press release | 17 | 16 | |
| `02-research` — dense academic prose | 18 | 14 | |
| `03-long-report` — long mixed report | 23 | 23 | |
| `04-code-claims` — prose *about code* | 16 | 8 | capability probe |
| **Total** | **74** | **61** | **82%** |

Two honest caveats, because they cut both ways:

- **My list is a floor, not the truth.** On `01-news` the model returned 19 facts against my
  17 — some of what it found beyond my list is perfectly good, and it scores no credit for
  it. Extra facts are reported but never graded.
- **Where it loses, it loses badly.** `02-research` is dense clinical prose, and the model
  drops whole clauses. `04-code-claims` is the interesting one: it reads prose claims about
  code fine, then skims the code itself. Every code span it found sits in a two-line block;
  every one it missed sits in a block of seven lines or more.

So: better than me at long, well-structured reports. Reliably worse on dense academic prose,
and clearly worse at reading code. **`--benchmark` exits non-zero on purpose** — two cases
fail today and are meant to stay failing until a model does better.

## The part that makes it a fair contest

Every `verbatim` span is **checked against the source before it is written**. A span the
model nearly copied is snapped to the exact source characters; a span not in the source at
all becomes `null`. After that pass, `verbatim` is either `null` or a real substring of your
input — never something the model believes it quoted.

That check is the product. The model is replaceable; the guarantee is not.

```json
{
  "facts": [
    {
      "id": 1,
      "fact": "Nordwind Energie will build a 480 MW offshore wind farm in the German Bight.",
      "type": "numeric",
      "confidence": "high",
      "verbatim": "it will build a 480 MW offshore wind farm in the German Bight"
    }
  ]
}
```

Three output states are worth telling apart:

| `verbatim` | `confidence` | Meaning |
|---|---|---|
| a span | any | Verified against the source |
| `null` | `low` | Inferred from the text, never claimed as a quote |
| `null` | `high` / `medium` | **A citation that failed verification** |

That last row is a feature, not a bug. It marks where the model *thought* it was quoting and
was wrong — the single most useful signal when comparing two models.

---

# Running it yourself

## Tested environment

Everything below was measured on one machine. Treat it as the known-good configuration, not
as a minimum.

| | |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop, driver 591.66, CUDA |
| CPU | AMD Ryzen 5 7235HS, 8 cores |
| RAM | 15.7 GB |
| Model | Qwen2.5-7B-Instruct-1M, Q4_K_M GGUF (4.68 GB) |
| Throughput | ~32 tok/s generation, model fully GPU-resident |

**6 GB of VRAM is the constraint that shapes every other number in this project.** One 7B
model at Q4_K_M fits with roughly 1 GB spare for the KV cache. There is no configuration
where two models are loaded at once.

## What you need

- **[Ollama](https://ollama.com) installed.** It does *not* need to be running — the tool
  starts `ollama serve` itself when nothing answers, applies the tuning environment from
  `settings.json`, and stops the whole process tree on exit (Ctrl-C included). A service you
  already have running is adopted as-is and left alone.
- **[Go](https://go.dev) 1.24+** to build. No external dependencies — standard library only.
- **An NVIDIA GPU with a CUDA-capable driver.** It will run CPU-only, but a 7B model on CPU
  is slow enough to make the benchmark impractical.
- **About 10 GB of free disk** for the model: the GGUF itself (~4.7 GB) plus the copy
  `ollama create` writes into its blob store (~4.7 GB again).
- **PowerShell** for `build.ps1`.

**Portability.** The Go code carries a POSIX path (`internal/service/proc_other.go` uses
process groups where Windows uses `taskkill /T`), so the binary should build on Linux and
macOS. That is untested, and `build.ps1` would need a shell equivalent. Windows is the
supported target.

## Getting the model

The build looks for the GGUF in two places, in order:

1. `models\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` next to the repo
2. a hardcoded development path

Download `Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` from
[lmstudio-community](https://huggingface.co/lmstudio-community) and drop it in `models\`, or
edit `$ggufCandidates` at the top of `build.ps1` to point wherever you keep it. If no GGUF
is found the build still succeeds — it skips model creation and prints the paths it looked
in. (If Ollama simply isn't on `PATH`, it prints the `ollama create` command to run later.)

**Using a different model?** Point `$ggufCandidates` at it and check its trained context
with `ollama show <model>` afterwards. A model trained below `num_ctx` (16384) degrades
silently rather than refusing.

## Setup

```powershell
.\build.ps1          # builds dist\, creates the Ollama model, writes settings.json
cd dist
```

`build.ps1` starts Ollama if it isn't running (`ollama create` needs a live service) and
stops it again afterwards — but only if the script was the one that started it.

Put your text in `prompt.md` and run:

```powershell
.\fact-extractor.exe
```

Confirm the model is fully on the GPU. **A partially offloaded model runs at roughly half
speed and reports no error anywhere**, so this check is not optional if you care about the
numbers:

```powershell
ollama ps            # PROCESSOR must read 100% GPU
```

## Adapting to your hardware

Everything tunable lives in `dist\settings.json`, regenerated by every build — so make
lasting changes in `build.ps1`, which writes it.

| Your VRAM | What to change |
|---|---|
| **6 GB** | Nothing. This is the tested configuration. |
| **8–12 GB** | `num_ctx` can go to 32768. Still one 7B model at a time. |
| **16 GB+** | A larger model becomes viable — update `$ggufCandidates` and re-check `ollama show`. |
| **4 GB** | Drop `num_ctx` to 8192 (saves ~246 MiB) and expect a tighter fit; consider a smaller quant. |
| **No NVIDIA GPU** | Set `num_gpu` to 0 to run on CPU. Correct, but slow. |

Three settings are **service environment variables**, not request options, and the VRAM
budget depends on them:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0     # halves KV cache vs fp16
OLLAMA_NUM_PARALLEL=1
```

The CLI applies these itself when it starts the service. If you run Ollama yourself, or set
`service.manage: false`, **you must set them by hand** — see
[`hardware-finetune.md`](hardware-finetune.md) §4.

## Long documents

Input is split on paragraph boundaries, then sentences, then a hard split; the resulting
fact lists are merged with duplicates dropped and ids renumbered.

`chunk_tokens` is the only knob trading speed against missed facts. Lower it to catch more,
raise it to go faster. **Raising `num_ctx` does not license raising it** — a bigger context
window is room for the model to *write* a long fact list, not licence to feed it a longer
passage to *read*.

## The corpus

Each case in `corpus/` pairs a source document with the facts I extracted from it by hand,
each anchored to an exact source span. A case passes with at most **3** gold facts unmatched
— run-to-run variance is the nature of the model, and 17 facts where a previous run found 18
is not a defect. Losing four or more means the model stopped extracting, which is a real and
otherwise silent failure.

`04-code-claims` is the odd one out: a **mixed document** of prose and comments making claims
about code, with the code alongside. Its gold list is eight planted claim/evidence pairs,
half of which contradict each other — a comment promising something the code below it does
not do. It measures whether both halves reach you with verified citations. The tool never
claims two facts disagree; deciding that is your job.

**Want to add your own case?** Write the document, then a `<name>.gold.json` beside it
listing the facts you found and the exact span supporting each. Two rules: every gold span
must be a real substring of the source (`go test ./internal/benchmark/` enforces this), and
avoid facts about an *absence* — they have no span to cite, so they cannot be graded.

## Validating output independently

`checkfacts` re-checks a `result.json` against its source without trusting the tool that
produced it — schema, ids, category vocabulary and verbatim traceability. It shares no code
with the extractor, deliberately, so that it can catch the extractor's bugs rather than
confirm them:

```powershell
.\checkfacts.exe --source prompt.md result.json
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ollama ps` shows less than 100% GPU | Partial CPU offload — roughly half speed, no error raised. Check the three env vars above, or lower `num_ctx`. |
| `has no model named "fact-extractor"` | The model was never created, or was created in a different store. Re-run `build.ps1`; check `OLLAMA_MODELS`. |
| `input does not fit` | The refusal is deliberate and happens before any request. Lower `chunk_tokens` to the value the message suggests. |
| Output cut off, `result.raw.txt` written | Generation hit the token cap. Lower `chunk_tokens`. |
| Extraction runs for many minutes | Some models never emit a stop token. There is no `num_predict` cap in `settings.json` — add one if you are experimenting with unfamiliar models. |
| `--benchmark` exits 1 | Expected. Two cases fail by design; see the scoreboard. |

## Documentation

- **[`AGENTS.md`](AGENTS.md)** — what this project is, the architecture, and the rules for
  changing it. The specification.
- **[`hardware-finetune.md`](hardware-finetune.md)** — every number that exists because of
  this specific GPU and model: measured VRAM budget, context scaling, sampling, and the
  Ollama service settings the budget depends on.

## Built on

This project is a thin thing standing on some very substantial ones.

- **[Ollama](https://ollama.com)** — model serving, the local registry, and the HTTP API
  this tool talks to. Its `format` field is what forwards a JSON Schema down to the engine.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the inference engine inside
  Ollama, and the GBNF grammar machinery that makes schema-constrained JSON output a
  guarantee rather than a request politely worded in a prompt.
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team at Alibaba — the model
  doing the actual work, in its 7B-Instruct-1M variant.
- **[lmstudio-community](https://huggingface.co/lmstudio-community)** — the GGUF
  quantization used here.
- **[Go](https://go.dev)** — the whole CLI is standard library. No dependency tree, one
  static binary.
- **llmfit** — used to estimate model/hardware fit while working out the VRAM budget in
  `hardware-finetune.md`. Its scores are pessimistic against measured `nvidia-smi` figures
  on this GPU, which the document records.
