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

## The experiment

Five documents, each read and extracted by hand *first* — the gold list is what I found,
every entry anchored to an exact source span. The model then sees the same document, the
same [`system-instruction.md`](system-instruction.md), greedy decoding, chunked at 1000
estimated tokens. One run per case, scored against my list.

### The scoreboard

| Document | My facts | Model found | |
|---|---:|---:|---|
| `01-news` — press release | 17 | 16 | |
| `02-research` — dense academic prose | 18 | 14 | |
| `03-long-report` — long mixed report | 23 | 23 | |
| `04-code-claims` — prose *about code* | 16 | 8 | capability probe |
| `05-survey` — dense multi-source literature review | 23 | 13 | multi-chunk density probe |
| **Total** | **97** | **74** | **76%** |

Measured at `chunk_tokens = 1000`.

#### The model

| | |
|---|---|
| Model | `Qwen2.5-7B-Instruct-1M`, Q4_K_M GGUF (~4.68 GB) |
| Served by | Ollama (llama.cpp, CUDA), `POST /api/chat`, `keep_alive: 0` |
| Decoding | greedy — `temperature 0.0`, `top_k 1`, `repeat_penalty 1.0`, `seed 42` |
| Context / chunk | `num_ctx 16384` · `chunk_tokens 1000` |
| System instruction | [`system-instruction.md`](system-instruction.md) — the exact prompt used for these numbers |
| Output contract | [`schemas/facts.json`](schemas/facts.json), sent as `format` (GBNF-constrained) |
| Throughput | ~32 tok/s generation, model fully GPU-resident |

The system instruction is linked because it's half the experiment: one edit to its
extraction rules and every number above stops being comparable to a re-run.

### Document by document

- **`01-news` — 16/17.** The model actually returned 19 facts against my 17, and still lost
  a point: two of the 19 restate the same claim in different words. A duplicate can't be
  credited to a second gold fact — matching is one extracted span per gold fact, at most —
  so the repeat earns nothing and the fact it should have covered goes down as a miss.
  **This is expected behaviour on short, dense text like this one** (one chunk, and nearly
  every clause is fact-bearing), where restating a claim slightly differently is an easy
  trap. It isn't a reading failure.
- **`02-research` — 14/18.** Dense clinical prose, one chunk. What it drops is the
  *secondary* clause: a third of this gold list lives in appositives and trailing
  qualifiers — "recruited from four clinics in Ontario," "assigned in a 1:1 ratio," "2.4 kg
  and 2.1 kg respectively" — and the model states a sentence's headline claim and moves on
  rather than splitting out what rides alongside it. This is precisely the case the system
  instruction's "a sentence that names a place, a date and an actor yields at least three
  facts" rule is meant to prevent, and it's the case where that rule holds least.
- **`03-long-report` — 23/23.** The only clean sweep, and the longest narrative document in
  the corpus, split across two chunks. Headed sections, one claim per sentence, facts spread
  evenly through the text — the shape the model handles best, and, not coincidentally, the
  shape I'm worst at, because reading 862 words of report prose carefully all the way through
  is exactly the tedium a human skims past.
- **`04-code-claims` — 8/16.** A capability probe, not a gate. Eight planted claim/evidence
  pairs, half of which contradict each other — a comment promising something the code below
  it doesn't do. It reads prose claims about code fine, then skims the code itself: every
  evidence span it found sits in a two-line block; every one it missed sits in a block of
  seven lines or more. The tool never claims two facts disagree — surfacing both halves with
  verified citations is the job here; deciding whether they agree is the reader's.
- **`05-survey` — 13/23.** The hardest case, and the only one that splits into three chunks.
  Same skimming pattern as `04`, now applied to a tab-separated table pasted as plain text
  and a bibliography-style source list instead of code: it missed **both** gold facts drawn
  from the table, and most of the source list. It also produced far more failed citations
  here (21) than on any other case combined — dense, attribution-heavy, list-shaped prose is
  where the model both skims hardest and misquotes most.

### The verdict

So: better than me at long, well-structured reports. Reliably worse on dense academic prose,
worse at reading code, and worse again on dense multi-source citation lists and tables — the
non-narrative shapes, consistently. **`--benchmark` exits non-zero on purpose** — three cases
fail today and are meant to stay failing until a model does better.

## The guarantee that makes it a fair contest

Every `verbatim` span is **checked against the source before it is written**. A span the
model nearly copied is snapped to the exact source characters; a span not in the source at
all becomes `null`. After that pass, `verbatim` is either `null` or a real substring of your
input — never something the model believes it quoted.

That check is the product. The model is replaceable; the guarantee is not.

```json
{
  "summary": {
    "total": 17, "verbatim": 14, "inferred": 3, "failed_citations": 1, "unlocated": 0
  },
  "verbatim_facts": [
    {
      "id": 1,
      "fact": "Nordwind Energie will build a 480 MW offshore wind farm in the German Bight.",
      "type": "numeric",
      "confidence": "high",
      "verbatim": "it will build a 480 MW offshore wind farm in the German Bight",
      "position": { "line": 12, "column": 5 }
    }
  ],
  "inferred_facts": [
    { "id": 4, "fact": "The project is publicly funded.", "type": "causal", "confidence": "low" }
  ]
}
```

The cited claims and the uncited ones go in separate arrays, and every citation carries the
`line` and `column` where it starts, so you can jump straight to it in your input instead of
searching the document for the quoted string. Both are 1-based. Ids are not renumbered by the
split: they stay 1..N across both arrays, so an id still identifies a fact.

`position` is `null` only in one narrow case — a span that sits across a chunk boundary and
so cannot be placed in the whole document. The fact keeps its citation; only the position is
missing.

Three output states are worth telling apart:

| `verbatim` | `confidence` | Meaning |
|---|---|---|
| a span | any | Verified against the source |
| `null` | `low` | Inferred from the text, never claimed as a quote |
| `null` | `high` / `medium` | **A citation that failed verification** |

That last row is a feature, not a bug. It marks where the model *thought* it was quoting and
was wrong — the single most useful signal when comparing two models.

## The environment it ran in

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

Every number that follows from that constraint — measured VRAM against context, the two
silent failure modes that halve throughput with no error raised, every request option and
service setting and why it's set that way, per-VRAM guidance for other hardware, and what
better hardware or a reasoning model would change — lives in
[`hardware-finetune.md`](hardware-finetune.md), not here. This file only reports the result
of running with those numbers; that one owns them.

## How the benchmark works

Each corpus case is a `<name>.gold.json` beside its source document:

```json
{
  "source": "corpus/01-news.md",
  "gold": [
    { "verbatim": "Twelve of the 74 stations", "type": "numeric" }
  ]
}
```

- **Matching.** Every gold span is first resolved against the real source. A gold entry that
  isn't a substring of its own source is a mistake in the corpus, not the model, and is
  reported as such rather than counted as a miss. A gold fact is then found if any extracted
  span contains it, or is contained by it — the model may cite a tighter clause than the gold
  list names, or a wider one that swallows it, and both mean it caught the fact. Because
  `verbatim` is by construction either `null` or a real substring of the source, this
  containment check is character-exact; there is no fuzzy matching anywhere in the pipeline.
- **One-to-one, and why it matters.** Each extracted span may account for at most one gold
  fact. Without that rule a single wide span — a whole sentence — could satisfy several gold
  facts at once: an early version of the benchmark reported 18 of 18 gold facts matched from
  an extraction that held only 15 facts in total. This same rule is what makes the `01-news`
  duplicate above cost a point instead of being credited twice.
- **Scoring.** 0–3 unmatched gold facts → **PASS**. 4 or more → **FAIL**. The tolerance
  exists because run-to-run variance is the nature of the model — 17 facts where a previous
  run found 18 is a near-tie falling the other way, not a defect. Losing four or more means
  the model stopped extracting partway through, which is a real and otherwise silent failure.
- **Reported, never scored.** Extracted count, facts found beyond my list, type mismatches
  against the gold's stated type, spans snapped to the source during verification, inferred
  facts (`verbatim: null` — no span, so nothing to match against), and citations that failed
  verification outright.
- **Preflight.** Before scoring, `--benchmark` warms the model and checks GPU placement
  (`checkPlacement` in `main.go`), warning loudly if it isn't 100% GPU-resident — a
  CPU-offloaded run is roughly half speed and would be measuring the wrong thing entirely.

### The second opinion: `checkfacts`

[`cmd/checkfacts/main.go`](cmd/checkfacts/main.go) exists to answer one question without
trusting the tool that produced the answer: **is this `result.json` actually what it claims
to be?** `--benchmark` runs it — via the same package, `internal/validate` — on every case
right after scoring it, printing `validation: contract OK (n/m spans exact)` or the list of
contract violations. It is **reported alongside the score, never part of pass/fail**; a
benchmark case can score well and still fail validation if the shape of the output is wrong.

`internal/validate` **shares no code with `internal/facts`**, deliberately — a validator
built out of the extractor's own helpers would just confirm `verify.go`'s bugs instead of
catching them. `checkfacts` is the thin file-reading, printing, exit-code shell around that
same independent package, which means the check the benchmark makes internally and the check
you can run by hand on any `result.json` are exactly the same check:

```powershell
.\checkfacts.exe --source prompt.md result.json
```

It checks schema, id continuity, the closed category vocabulary, that every `verbatim` is a
real substring of the source, and that each reported `line`/`column` actually lands on its
span — re-derived independently rather than trusted from the output. A position that points
at the wrong line is a **failure, not a warning**: it reads as authoritative while sending
you somewhere the quote is not.

---

# Running it yourself

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

Everything tunable lives in `dist\settings.json`, regenerated by every build. Per-VRAM
guidance for hardware other than the tested 6 GB card, and the service environment variables
the VRAM budget depends on, are in
[`hardware-finetune.md`](hardware-finetune.md) §2.9 and §2.3.

## The corpus

Each case in `corpus/` pairs a source document with the facts I extracted from it by hand,
each anchored to an exact source span. Why each one is in the suite:

- **`01-news`** — the baseline. Short and dense enough that nearly every clause is
  fact-bearing, and it fits in a single chunk, which isolates raw extraction quality from any
  effect of chunking or merging.
- **`02-research`** — hedges, modality, and subordinate clauses. It tests whether qualifiers
  survive the trip: a fact stated as requested, hedged, or approximate must not come out
  reading as certain or complete.
- **`03-long-report`** — the multi-chunk narrative case. It checks whether merging across a
  chunk boundary loses or duplicates facts, and whether citation positions still resolve
  correctly against the whole document rather than just the chunk they were found in.
- **`04-code-claims`** — a mixed document of prose and comments making claims about code,
  with the code alongside. Its gold list is eight planted claim/evidence pairs, half of which
  contradict each other. It measures whether both halves reach you with verified citations;
  the tool never decides that two facts disagree — that's your job.
- **`05-survey`** — the density probe. Attribution-heavy prose, a plain-text table, and one
  paragraph long enough to force the sentence-level fallback in `textsplit` rather than
  staying a single paragraph unit. At `chunk_tokens = 1000` it's the only document in the
  corpus that splits into three chunks.

A case passes with at most **3** gold facts unmatched — run-to-run variance is the nature of
the model, and 17 facts where a previous run found 18 is not a defect. Losing four or more
means the model stopped extracting, which is a real and otherwise silent failure.

**Want to add your own case?** Write the document, then a `<name>.gold.json` beside it
listing the facts you found and the exact span supporting each. Two rules: every gold span
must be a real substring of the source (`go test ./internal/benchmark/` enforces this), and
avoid facts about an *absence* — they have no span to cite, so they cannot be graded.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ollama ps` shows less than 100% GPU | Partial CPU offload — roughly half speed, no error raised. Check the three env vars in `hardware-finetune.md` §2.3, or lower `num_ctx`. |
| `has no model named "fact-extractor"` | The model was never created, or was created in a different store. Re-run `build.ps1`; check `OLLAMA_MODELS`. |
| `input does not fit` | The refusal is deliberate and happens before any request. Lower `chunk_tokens` to the value the message suggests. |
| Output cut off, `result.raw.txt` written | Generation hit the token cap. Lower `chunk_tokens`. |
| Extraction runs for many minutes | Some models never emit a stop token. There is no `num_predict` cap in `settings.json` — add one if you are experimenting with unfamiliar models. |
| `--benchmark` prints a `WARNING: the model is only ...` placement notice | The preflight check found less than 100% GPU residency; the run that follows isn't comparable to the recorded numbers. Same fix as the first row. |
| `--benchmark` exits 1 | Expected. Three cases fail by design; see the scoreboard. |

## Documentation

- **[`AGENTS.md`](AGENTS.md)** — what this project is, the architecture, and the rules for
  changing it. The specification.
- **[`hardware-finetune.md`](hardware-finetune.md)** — every number that exists because of
  this specific GPU and model: measured VRAM budget, context scaling, sampling, and the
  Ollama service settings the budget depends on.
- **[`system-instruction.md`](system-instruction.md)** — the exact prompt behind every score
  above. Editing it is what most changes the numbers; diff it before you diff anything else.

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
- **[llmfit](https://github.com/AlexsJones/llmfit)** — used to estimate model/hardware fit
  while working out the VRAM budget in `hardware-finetune.md`. Its scores are pessimistic
  against measured `nvidia-smi` figures on this GPU, which the document records.

## Where this goes next

The losses above aren't random. They cluster on text that isn't narrative — code blocks, a
table, a bibliography-style source list, subordinate clauses — and the failure mode is
always the same shape: the model stops early and hands back a valid but incomplete list.
Nothing forces it to keep going. The grammar guarantees shape, never exhaustiveness — `]` is
a legal token after any finished fact object — and no sampling parameter available today
addresses that (`hardware-finetune.md` §2.8).

Two directions look more promising than tuning this setup further:

- **A reasoning-capable model.** Given a thinking budget, a model could plan something like a
  sentence-by-sentence sweep and check its own coverage before it commits to closing the
  array — attacking the "the grammar permits stopping" cause directly, which no sampling
  parameter can touch. The honest cost: thinking tokens compete for the same output headroom
  `num_ctx 16384` exists to buy, on a card that generates at 25–32 tok/s to begin with
  (`hardware-finetune.md` §3).
- **A model trained for this task, not for chat.** An instruct model is tuned to be helpful
  and concise; this job wants exhaustive, and those two objectives pull against each other on
  every dense paragraph in this corpus. The data shape that actually matches is *claim plus
  exact supporting span*, which is what
  [FEVER](https://github.com/QiangAIResearcher/Fact-Extraction-and-Verification) (Thorne et
  al., NAACL 2018 — 185,441 claims checked against Wikipedia) provides. Worth being precise
  about which half of it transfers: FEVER's **evidence-selection** stage — pick the sentence
  that supports a claim — is the same operation as citing a `verbatim` span, and training on
  it directly targets the recall problem here. Its **claim-classification** stage (Supported
  / Refuted / NotEnoughInfo) makes it a fact *checker*, which this project deliberately isn't
  — though the contradicting pairs planted in `04-code-claims` sit right next door to that
  line. Open information extraction corpora (CaRB, LSOIE) are a closer match for exhaustive,
  clause-level extraction specifically, and SciFact is worth a look for the dense scientific
  prose that `02-research` loses on.

Neither of these happens on this machine — 6 GB won't hold even a QLoRA run over a 7B model,
so fine-tuning would mean renting a GPU elsewhere. And the corpus in `corpus/` is 97
hand-written facts, sized to be a benchmark, not a training set; it should stay held out
rather than folded into training data for whatever comes next. Whichever direction gets
tried, it gets scored exactly the way the model above was: the same gold list, the same
one-to-one matching, the same verification pass. `--benchmark` is the gate either way.
