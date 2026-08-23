# fact-extractor

A Windows CLI that turns one text file into a structured, **source-traceable** list of
facts, and exits.

```powershell
fact-extractor.exe                      # prompt.md -> result.json
fact-extractor.exe --benchmark          # score the gold corpus, exit 0 or 1
fact-extractor.exe --benchmark --coder  # score the same corpus under a second model
```

There are no other flags. Model, context and sampling come from `settings.json`, which the
build generates.

## What makes it different from asking a model for facts

Every `verbatim` span in the output is **checked against the source before it is written**.
A span the model nearly copied is snapped to the exact source characters; a span that is
not in the source at all becomes `null`. After that pass, `verbatim` is either `null` or a
real substring of your input — never something the model believes it quoted.

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

Reading the output, three states are worth knowing apart:

| `verbatim` | `confidence` | Meaning |
|---|---|---|
| a span | any | Verified against the source |
| `null` | `low` | Inferred from the text, never claimed as a quote |
| `null` | `high` / `medium` | **A citation that failed verification** |

## Requirements

- **Ollama installed.** It does not need to be running: the tool starts `ollama serve`
  itself when none is answering — with the tuning environment from `settings.json`
  applied — and stops it again on exit. A service you already have running is used as-is
  and left alone.
- An NVIDIA GPU with a CUDA-capable driver. Developed against an RTX 3050 6 GB.
- Go 1.24+ to build. No external Go dependencies.

If you run Ollama yourself instead (or set `service.manage: false`), three settings the
VRAM budget depends on must be set on that service by hand — see
[`hardware-finetune.md`](hardware-finetune.md) §4.

## Setup

```powershell
.\build.ps1          # builds dist\, creates the Ollama model, writes settings.json
cd dist
```

Then put your text in `prompt.md` and run it:

```powershell
.\fact-extractor.exe
```

Confirm the model is fully on the GPU — a partially offloaded model is roughly half speed
and reports no error:

```powershell
ollama ps            # PROCESSOR must read 100% GPU
```

## Benchmark mode

`--benchmark` answers one question: **did a new model, or an edited
`system-instruction.md`, make extraction worse?**

Each case in `corpus/` pairs a source text with a hand-authored list of facts that must
come out of it, each anchored to a source span. A case passes with at most **3** gold facts
unmatched, because run-to-run variance is the nature of the model — 17 facts where a
previous run found 18 is not a defect. Losing four or more means the model stopped
extracting, which is a real and otherwise silent failure.

```powershell
.\fact-extractor.exe --benchmark
```

It also reports, without scoring them, how many citations failed verification. That number
is the one to watch when comparing two models.

One case, `04-code-claims`, is a **mixed document**: prose and comments making claims about
code, with the code alongside. Its gold list is eight planted claim/evidence pairs, half of
which contradict each other. It measures whether both halves reach you with verified
citations — the tool never claims two facts disagree, because deciding that is your job.

`--coder` runs the same corpus under `ollama.coder_model`, so a second model can be scored
against the first on identical inputs. Only one model is ever named per run, so both are
never resident at once.

## Long documents

Input is split on paragraph boundaries, then sentences, then a hard split, and the
resulting fact lists are merged with duplicates dropped and ids renumbered.

`chunk_tokens` in `settings.json` is the only knob that trades speed against missed facts.
Lower it to catch more; raise it to go faster. **Raising `num_ctx` does not license raising
it** — a bigger context window is room for the model to *write* a long fact list, not
licence to feed it a longer passage to *read*.

## Validating output independently

`checkfacts` re-checks a `result.json` against its source without trusting the tool that
produced it — schema, ids, category vocabulary and verbatim traceability:

```powershell
.\checkfacts.exe --source prompt.md result.json
```

## Documentation

- **[`AGENTS.md`](AGENTS.md)** — what this project is, the architecture, and the rules for
  changing it. The specification.
- **[`hardware-finetune.md`](hardware-finetune.md)** — every number that exists because of
  this specific GPU and model: VRAM budget, context, sampling, and the Ollama service
  settings the budget depends on.

## What it is not

Not an agent — no loop, no tool use, no autonomy; control flow is fixed before the model is
called. Not a general LLM runner. Not a fact *checker*: it extracts what a text claims and
proves the citation, and takes no position on whether the text is right.
