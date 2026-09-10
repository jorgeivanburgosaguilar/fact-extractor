# fact-extractor

**A toy project asking one question: is a local LLM better than I am at pulling the facts
out of a piece of text?**

Toy in scope, not in method. To answer that honestly you need a fair contest, so the
project is really three things: a CLI that makes a small local model extract facts and
*proves* every citation against the source, a corpus of documents I read and extracted by
hand, and a benchmark that scores a model against my answers.

```powershell
fact-extractor.exe                                # prompt.md -> result.json (Gemma 4, thinking on by default)
fact-extractor.exe --benchmark                     # score the default model
fact-extractor.exe --model fact-extractor-qwen     # use the historical Qwen baseline instead
```

Context and sampling come from `settings.json`, which the build generates. `--model`
overrides which Ollama model a run uses (the default still comes from `settings.json`);
a reasoning model thinks by default, and `--no-think` turns that off. Neither touches
`settings.json` on disk.

**The default model is always the current benchmark winner.** Whichever model matches more
gold facts in total — see the scoreboard below — is the one `fact-extractor.exe` uses with
no `--model` flag. A model that loses that comparison isn't dropped: it stays reachable with
`--model` as a historical baseline, so it can be re-scored the next time
`system-instruction.md` or the corpus changes. Right now that's **Gemma 4 E2B** (non-QAT) as
the default, with **Gemma 4 E2B QAT** (`fact-extractor-gemma4-qat`) and
**Qwen2.5-7B-Instruct-1M** (`fact-extractor-qwen`) as the two baselines it replaced — see
[`AGENTS.md`](AGENTS.md) §2.

## The experiment

Five documents, each read and extracted by hand *first* — the gold list is what I found,
every entry anchored to an exact source span. Each model then sees the same document, the
same [`system-instruction.md`](system-instruction.md), greedy decoding, chunked at 1000
estimated tokens. One run per case, scored against my list.

Three models have been run through this so far: **Qwen2.5-7B-Instruct-1M**, the original
baseline; **Gemma 4 E2B QAT**, a much smaller reasoning model run with thinking on by
default; and **Gemma 4 E2B** (non-QAT), the same architecture quantized the ordinary way
instead of QAT's quantization-aware training — a genuinely different weight set, not a
different quant scheme of the same checkpoint. The system instruction is identical across
all three — it's half the experiment, and editing it would make every number below
incomparable to a re-run. Two separate comparisons live in these columns: Qwen vs. Gemma QAT
isolates **model + thinking budget** together (see
[The reasoning-model result](#the-reasoning-model-result)); Gemma QAT vs. non-QAT Gemma
isolates **just the weights**, thinking held constant on both (see
[The QAT vs. non-QAT result](#the-qat-vs-non-qat-result)).

### The scoreboard

| Document | My facts | Qwen2.5-7B | Gemma 4 QAT (think) | Gemma 4 (think) |
|---|---:|---:|---:|---:|
| `01-news` | 17 | 16 | 16 | **16** |
| `02-research` | 18 | 14 | 18 | **18** |
| `03-long-report` | 23 | **23** | 22 | 22 |
| `04-code-claims`* | 16 | 8 | 11 | **11** |
| `05-survey`* | 23 | 13 | 16 | **18** |
| **Total** | **97** | 74 (76%) | 83 (86%) | **85 (88%)** |

\* capability probe, not a gate — see [Document by document](#document-by-document) below.

Measured at `chunk_tokens = 1000`. Bold marks the single best model on that document; a tie
goes to the faster one (`01-news`: all three find 16, non-QAT Gemma's ~69 tok/s beats QAT's
~58 and Qwen's ~32; `02-research` and `04-code-claims` are two-way ties between the Gemma
builds, same tie-break) — a benchmark exists to catch regressions, and equal recall at lower
wall-clock cost is a real win, not a wash, so the scoreboard never leaves a tie unresolved.
See `AGENTS.md` §5. Non-QAT Gemma is the current benchmark winner and backs the default
`fact-extractor` name (`AGENTS.md` §2); the other two are historical baselines.

#### The models

| | Qwen2.5-7B-Instruct-1M | Gemma 4 E2B QAT | Gemma 4 E2B (non-QAT) |
|---|---|---|---|
| File | Q4_K_M GGUF, 4.68 GB (4683073888 bytes) | Q4_0 GGUF, 3.35 GB (3349515424 bytes) | Q4_K_M GGUF, 3.43 GB (3427880384 bytes) |
| Served by | Ollama (llama.cpp, CUDA), `POST /api/chat`, `keep_alive: 0` | same | same |
| Decoding | greedy — `temperature 0.0`, `top_k 1`, `repeat_penalty 1.0`, `seed 42` | same | same |
| Context / chunk | `num_ctx 16384` · `chunk_tokens 1000` | same | same |
| Thinking | not applicable | **on** (the CLI's default; `--no-think` would turn it off) — Ollama's `gemma4` parser splits the trace into `message.thinking`; `message.content` stays clean JSON, verified for every run in this table | same — identical `capabilities` reported by `/api/show` |
| System instruction | same — [`system-instruction.md`](system-instruction.md), unedited between runs | same | same |
| Output contract | same — [`schemas/facts.json`](schemas/facts.json), sent as `format` (GBNF-constrained), honoured with thinking on | same | same |
| Throughput | ~32 tok/s generation, fully GPU-resident | ~58 tok/s generation, fully GPU-resident, but thinking tokens and content tokens draw from the *same* output budget — see below | ~69 tok/s generation, fully GPU-resident, same shared-budget caveat |
| VRAM (whole system, `num_ctx 16384`) | 5587 MiB (measured, `hardware-finetune.md` §1.5) | 2161 MiB (measured) | 2536 MiB (measured, `hardware-finetune.md` §1.9) |

The system instruction is linked because it's half the experiment: one edit to its
extraction rules and every number above stops being comparable to a re-run. It was **not**
adapted for Gemma's thinking — see the note at the end of this section. QAT and non-QAT are
genuinely different weight sets (QAT is fine-tuned for quantization robustness; non-QAT is
the base release quantized the ordinary way), not two quantizations of one checkpoint, so
neither row above was assumed to carry over from the other — both were measured directly.

### The reasoning-model result

Gemma's thinking trace costs real tokens: on `01-news` alone it runs 6364–7203 characters
per chunk, and the five-case benchmark spent about 54,000 characters of reasoning in total
against 233 raw extracted facts. `eval_count` on a single chunk went from 1452 tokens with
thinking off to 3442 with it on — roughly **2.4×** the generation cost for that one
chunk, confirmed with a direct `think:true` vs `think:false` probe against the live model
before any benchmark ran (`hardware-finetune.md` §3 has the full comparison). That is the
predicted cost `hardware-finetune.md` named before this model was ever run: thinking and
content tokens share one output budget, with no separate accounting.

What that spend buys, on this evidence: `02-research` — Qwen's worst case, losing a third of
its gold list to subordinate clauses and qualifiers — goes to a **clean 18/18 sweep** with
thinking on. `04-code-claims` and `05-survey`, the two capability probes that are *meant* to
stay hard, both improve (8→11, 13→16) without closing entirely. Citation quality also moves:
Gemma's total failed-citation count across all five cases is 8, against Qwen's 21 on
`05-survey` alone. The one case that gets slightly worse is `03-long-report` (23→22), losing
a single figure (`£46,000 per station`) that Qwen caught — not a pattern, on this evidence.

None of this isolates *thinking* from *model* — a smaller model with a different
architecture is also a variable — but the composition (thinking on, same day, same
hardware, same prompt, same schema) is exactly the reasoning-capable-model experiment this
project's "Where this goes next" section named as the obvious next step, back when the
answer was still a prediction rather than a measurement.

**On `system-instruction.md`:** before deciding whether to adapt it for a reasoning model, a
direct probe sent the real system instruction, the real schema, and a real corpus document
to Gemma three ways — `think` omitted, `true`, `false` — and read back `message.thinking`
and `message.content` separately. The reasoning trace never once leaked into `content`;
`content` was clean, schema-valid JSON in every case. On that evidence the instruction was
left frozen: there was no technical problem to fix, and editing it would have made the Qwen
column above incomparable, forcing a re-run of a model whose numbers already stand.

### The QAT vs. non-QAT result

This comparison holds thinking constant — both Gemma builds run with it on by default — and
isolates the one variable the name suggests: the weights. QAT (quantization-*aware*
training) fine-tunes a checkpoint to be robust under quantization; the non-QAT release here
is the ordinary base checkpoint quantized the regular way, to the same Q4-class precision.
Different training objective, different resulting weights, not a repackaging of the same
file.

Non-QAT wins outright: **85/97 (88%)** against QAT's **83/97 (86%)** — a clean win on
`05-survey` (18/23 against 16/23) and an exact tie on the other four documents, never a
regression. It's also faster (~69 tok/s against ~58) and its thinking traces run shorter per
case (3151–15921 characters against QAT's 4502–20556) — the full five-case benchmark
finished in 391.8 s against 470.5 s for QAT in the same session. None of that was
predictable from `general.size_label` or the GGUF metadata, which are identical between the
two (`hardware-finetune.md` §1.9 has the byte-for-byte comparison); it only shows up once
both are actually run against the corpus.

`llmfit` was checked before either GGUF was pulled into Ollama — its database has no entry
for the QAT release specifically (only community re-quants surface), but its canonical
`google/gemma-4-E2B-it` entry gave a "Good" fit verdict for this card at `num_ctx 16384`,
same ranking-level reassurance §1.3 already established `llmfit` is good for on this project,
while overestimating the actual VRAM cost by roughly the same margin measured there
(`hardware-finetune.md` §1.9 has the numbers).

### Document by document

#### Qwen2.5-7B-Instruct-1M

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

#### Gemma 4 E2B QAT, thinking on

- **`01-news` — 16/17.** Same score as Qwen, different miss: it dropped "founded in 1984 in
  Bremerhaven" rather than losing a point to a near-duplicate. One classified fact and one
  failed citation on an otherwise clean single chunk.
- **`02-research` — 18/18.** A clean sweep on exactly the document Qwen loses the most on.
  The qualifiers and appositives that Qwen folds into a headline claim — the case
  `system-instruction.md`'s "three facts per sentence" rule targets and, per the Qwen
  write-up above, holds least on — come out as separate facts here. The most direct evidence
  in this whole comparison that the thinking budget is buying a coverage sweep rather than
  just more tokens.
- **`03-long-report` — 22/23.** Nearly the same clean result as Qwen, missing only
  "Replacement cost is estimated at £46,000 per station" — a single figure, not a pattern.
- **`04-code-claims` — 11/16.** Still the harder of the two capability probes, and still
  failing by the same shape Qwen fails by: the five misses are all evidence spans sitting
  inside the code blocks (an `fmt.Errorf` call, a doc comment, two `return` statements, an
  `EqualFold` call), while the prose halves of the same claim/evidence pairs come through.
  Better than Qwen's 8/16, not a different failure mode.
- **`05-survey` — 16/23.** Still the hardest case by a wide margin, and still missing from
  the same two places: the tab-separated table (misses every figure drawn from it) and the
  attribution-heavy interview/methodology prose around it. Genuinely better than Qwen's
  13/23, and with far fewer failed citations along the way (7 here against Qwen's 21) — but
  dense, list-shaped, non-narrative text is still where this comparison's advantage narrows
  the most.

#### Gemma 4 E2B (non-QAT), thinking on

- **`01-news` — 16/17.** Same score as both other models, a third different miss: it dropped
  "it will build a 480 MW offshore wind farm in the German Bight" rather than QAT's Bremerhaven
  founding date or Qwen's near-duplicate. One classified fact and one span snapped to the
  source on an otherwise clean single chunk.
- **`02-research` — 18/18.** The same clean sweep QAT gets on Qwen's worst case — the
  qualifiers-and-appositives pattern comes out as separate facts here too, so this isn't
  something QAT's training bought specifically; both Gemma builds do it.
- **`03-long-report` — 22/23.** The same score as QAT, a different single miss: "340
  kilometres of the eastern seaboard" instead of QAT's "£46,000 per station" — the two Gemma
  builds land on the same total by missing different things, not the same thing.
- **`04-code-claims` — 11/16.** The same score and the same failure shape as QAT: all five
  misses are evidence spans inside code blocks (`const DefaultPort = 7433`, an `fmt.Errorf`
  call, two `return` statements, an `EqualFold` call), prose halves of the same pairs come
  through clean. Neither Gemma build has cracked reading code past a couple of lines.
- **`05-survey` — 18/23.** The one case where this build pulls ahead of its own sibling, not
  just Qwen: it recovers two of QAT's seven misses — "214 Berthings at Sandhaven Container
  Terminal" and the "magnetic mooring aborted in only 4%..." comparison — while still missing
  the same tab-separated table figures and attribution-heavy interview prose both other models
  struggle with. Zero failed citations here, against QAT's 7 and Qwen's 21 on this document.

### The verdict

All three models are better than me at long, well-structured reports, and all three are
reliably worse on dense academic prose, code, and multi-source citation lists and tables —
the non-narrative shapes, consistently. What differs is *how much* worse: either Gemma build
with thinking closes Qwen's gap on dense academic prose entirely (`02-research`, 14/18 →
18/18) and narrows it meaningfully on the two shapes that are hardest by design
(`04-code-claims`, `05-survey`) — and between the two Gemma builds, non-QAT narrows
`05-survey` further still (16/23 → 18/23) on the same thinking budget, same corpus, same
everything but the weights. **`--benchmark` exits non-zero on purpose** for all three models
today — `04-code-claims` and `05-survey` fail under both Gemma builds too, and are meant to
stay failing until a model closes them, per `AGENTS.md` §5: they measure surfacing and
coverage under stress, not a bar any of them is expected to clear soon.

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

`build.ps1` creates **up to three** Ollama models, each independently optional — a missing
GGUF skips only that entry, never the build:

| Model | Default `.gguf` locations (tried in order) | Ollama name |
|---|---|---|
| Gemma 4 E2B, non-QAT (the default `settings.json` names — current benchmark winner) | `models\gemma-4-E2B-it-Q4_K_M.gguf` next to the repo, then a hardcoded development path | `fact-extractor` |
| Gemma 4 E2B QAT (historical baseline, reachable only via `--model`) | `models\gemma-4-E2B-it-QAT-Q4_0.gguf` next to the repo, then a hardcoded development path | `fact-extractor-gemma4-qat` |
| Qwen2.5-7B-Instruct-1M (historical baseline, reachable only via `--model`) | `models\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` next to the repo, then a hardcoded development path | `fact-extractor-qwen` |

Download `Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf`, `gemma-4-E2B-it-QAT-Q4_0.gguf`, and
`gemma-4-E2B-it-Q4_K_M.gguf` from
[lmstudio-community](https://huggingface.co/lmstudio-community) and drop them in `models\`,
or edit `$models` at the top of `build.ps1` to point wherever you keep them. Ignore the
sibling `mmproj-*.gguf` next to either Gemma download — that's the vision projector, unused
here. If a GGUF is missing the build still succeeds — it skips that model's creation and
prints the paths it looked in. (If Ollama simply isn't on `PATH`, it prints the
`ollama create` command to run later, one per resolved model.)

`$models[0]` is always the current benchmark winner, per [`AGENTS.md`](AGENTS.md) §2 — a
model that loses the comparison drops to a later entry rather than being removed, so it
stays buildable and re-scorable as a historical baseline.

**Using a different model?** Add an entry to `$models` in `build.ps1` and check its trained
context with `ollama show <model>` afterwards. A model trained below `num_ctx` (16384)
degrades silently rather than refusing. Run it with `--model <name>`; if it's a reasoning
model, thinking is on by default (`--no-think` turns it off) — but check first whether
Ollama's builtin parser actually
splits its thinking channel out of `message.content` (`hardware-finetune.md` §3 shows how
this project probed that for Gemma before trusting it), because a model whose thinking
leaks into content fails extraction with a named, saved-to-`result.raw.txt` error rather
than silently corrupting output.

## Setup

```powershell
.\build.ps1          # builds dist\, creates the Ollama models it finds GGUFs for, writes settings.json
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
| `--benchmark` exits 1 | Expected. Some cases fail by design under every model tried so far; see the scoreboard. |
| `has no model named "..."` after `--model <name>` | That model was never created — either its `.gguf` was missing at build time, or the name is misspelled. `Preflight` lists what's actually available; re-run `build.ps1` after fixing `$models` or getting the `.gguf`. |
| `--model` names a model whose thinking leaks into `message.content` | Extraction fails with a named error (not silent corruption) and saves the raw reply to `result.raw.txt`. Means Ollama's builtin parser for that architecture isn't splitting the reasoning channel out — check with a direct probe before trusting thinking-on-by-default on an unfamiliar model, the way `hardware-finetune.md` §3 did for Gemma 4. |

## Documentation

- **[`AGENTS.md`](AGENTS.md)** — what this project is, the architecture, and the rules for
  changing it. The specification.
- **[`hardware-finetune.md`](hardware-finetune.md)** — every number that exists because of
  this specific GPU and the models run on it: measured VRAM budget, context scaling,
  sampling, and the Ollama service settings the budget depends on.
- **[`system-instruction.md`](system-instruction.md)** — the exact prompt behind every score
  above. Editing it is what most changes the numbers; diff it before you diff anything else.

## Built on

This project is a thin thing standing on some very substantial ones.

- **[Ollama](https://ollama.com)** — model serving, the local registry, and the HTTP API
  this tool talks to. Its `format` field is what forwards a JSON Schema down to the engine.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the inference engine inside
  Ollama, and the GBNF grammar machinery that makes schema-constrained JSON output a
  guarantee rather than a request politely worded in a prompt.
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team at Alibaba — the original
  baseline model, in its 7B-Instruct-1M variant.
- **[Gemma](https://ai.google.dev/gemma)** by Google DeepMind — the reasoning-model
  comparison, in both its 4 E2B QAT and non-QAT variants; the non-QAT release is the current
  benchmark winner and backs the default model. Its chat template and thinking-channel
  protocol are what Ollama's `gemma4` builtin parser splits `message.thinking` from
  `message.content` on, for both variants alike.
- **[lmstudio-community](https://huggingface.co/lmstudio-community)** — the GGUF
  quantizations used here, for all three models.
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

**A reasoning-capable model was the first of two directions this project named as more
promising than tuning further — it has now been tried, not just predicted.** Gemma 4 E2B,
thinking on by default, closes `02-research` entirely (14/18 → 18/18) and narrows both remaining
capability probes (`04-code-claims` 8→11, `05-survey` 13→16) without closing them, on a model
under a third the size of Qwen. The mechanism looks like the one predicted: a thinking budget
appears to buy something closer to a coverage sweep before the array closes, rather than the
model taking the first list that satisfies the grammar — most visible on `02-research`, where
the qualifiers and appositives Qwen collapses into a headline claim come out as separate
facts instead. The honest cost also landed as predicted: thinking and content tokens draw
from the same output budget with no separate accounting, and it is a real cost — roughly
2.4× the generation tokens for a single chunk in a direct `think:true` vs `think:false`
probe (`hardware-finetune.md` §3) — not a rounding error. What is not yet known: whether this
generalizes past one small model and one document set, whether a larger reasoning model
would close the two remaining probes rather than merely narrow them, and how the thinking
cost scales on `05-survey`'s three-chunk case specifically, where it's most expensive. The
full comparison is in [the scoreboard above](#the-scoreboard).

**One of those open questions answered itself without a larger model.** Swapping the QAT
checkpoint for the non-QAT release of the same architecture — same thinking mechanism, same
everything else — narrowed `05-survey` further (16/23 → 18/23) and did it faster (~69 tok/s
against ~58), not at a speed cost. That is weights, not scale or thinking budget, doing the
work; see [The QAT vs. non-QAT result](#the-qat-vs-non-qat-result) above. It does not answer
whether a *larger* reasoning model would close the probes rather than narrow them, but it is
evidence that this particular gap is not purely a parameter-count story either.

A second direction remains untried:

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
  prose that Qwen loses on (`02-research`) — the case Gemma's thinking, above, already closes
  without any training at all.

This one does not happen on this machine — 6 GB won't hold even a QLoRA run over a 7B model,
so fine-tuning would mean renting a GPU elsewhere; unlike the reasoning-model comparison
above, which is inference-only and ran entirely on the hardware in `hardware-finetune.md`
§1. The corpus in `corpus/` is 97 hand-written facts, sized to be a benchmark, not a
training set; it should stay held out rather than folded into training data for whatever
comes next. Whichever model gets tried next, it gets scored exactly the way both models
above were: the same gold list, the same one-to-one matching, the same verification pass.
`--benchmark` is the gate either way.
