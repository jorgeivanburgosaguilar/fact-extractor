# AGENTS.md — fact-extractor

The single specification for this repository: what it is, why it exists, how it is built,
and the rules for changing it. There is no separate work plan.

Every number that exists because of this GPU and this model — context, VRAM, sampling,
runtime settings — lives in **[`hardware-finetune.md`](hardware-finetune.md)**. This file
never restates those figures; it references the section.

---

## 1. What this project is

A Windows CLI in **Go** (`fact-extractor.exe`) that turns one text file into a structured,
**source-traceable** list of facts, and exits.

```powershell
fact-extractor.exe                                # system-instruction.md + prompt.md -> result.json (Gemma 4, thinking on by default)
fact-extractor.exe --benchmark                     # score the gold corpus, exit 0 or 1
fact-extractor.exe --model fact-extractor-qwen     # the historical Qwen baseline instead
```

It does exactly one job. There is no profile system and no output format other than JSON.
The CLI takes exactly three flags: `--benchmark`, `--model` (name a different Ollama model
for this run) and `--no-think` (send `think: false` explicitly, overriding a reasoning
model's own default of thinking-on). Neither overrides more than the one setting it names —
`--model` overrides `ollama.model`, `--no-think` overrides the request's `think` field, and
both apply only for the run, never to disk. See
§2's Configuration section for the boundary that keeps this from becoming a profile system.
Context and sampling still come from `settings.json`; input and output filenames are fixed.

A second, much smaller binary ships alongside it: **`checkfacts.exe`** (built from
`cmd/checkfacts`), which independently re-validates a `result.json` against its source —
see §5. `build.ps1` builds both; `checkfacts.exe` is not something `fact-extractor.exe` can
be asked to do, and the two binaries deliberately share no validation code
(`internal/validate` vs. `internal/facts`).

### The identity to protect

**A fact extractor whose citations are traceable to the source.** Every `verbatim` span in
the output is either a real substring of the input or `null` — never something the model
merely believes it copied. That guarantee, not the model and not the runtime, is the
product.

It is **not an agent**: no loop, no tool use, no autonomy. Control flow is fixed before the
model is ever called.

### Model schema (frozen)

`schemas/facts.json` — `id`, `fact`, `type`, `confidence`, `verbatim`. Seven categories:
`quote`, `numeric`, `event`, `entity`, `definition`, `causal`, `other`.

The schema is a theory of what a fact is. Changing it invalidates every comparison between
models and between system instructions, so it does not change casually.

### On-disk shape (not the same thing)

`schemas/facts.json` is the `format` field sent to Ollama: the **model's** contract, and the
thing that is frozen. `result.json` is what a **person** opens, and it is allowed to be a
different shape, because rearranging it costs no comparability:

```json
{
  "summary": { "total": 17, "verbatim": 14, "inferred": 3,
               "failed_citations": 1, "unlocated": 0 },
  "verbatim_facts": [ { …, "verbatim": "…", "position": { "line": 12, "column": 5 } } ],
  "inferred_facts": [ { "id": 4, "fact": "…", "type": "causal", "confidence": "low" } ]
}
```

`internal/output` builds this after extraction and verification are over. It splits the facts
on whether `verbatim` is null and locates each citation in the source. Two rules make the
split safe to rely on:

- **Ids are never renumbered.** They stay 1..N across the *union* of both arrays, so an id
  remains the fact's identity.
- **`position` is derived here, never asked of the model.** It is `line` and `column`, both
  1-based, column counted in runes. It is `null` only when a span could not be located in the
  whole source — see the chunk-seam case in §4.

The pipeline itself still works on the flat list: `facts.Parse`, `facts.Verify` and
`facts.Merge` are untouched by any of this.

---

## 2. Architecture

```
[ settings.json ]──────────► model, options, chunk size
[ system-instruction.md ]──► system message  ─┐
[ prompt.md ]──────────────► user messages  ──┴──► [ fact-extractor.exe (Go) ]
                                                          │
                                                          │  POST /api/chat
                                                          │  { format, options, keep_alive: 0 }
                                                          ▼
                                                  [ Ollama (llama.cpp, CUDA) ]
                                                          │
                                          chunk → extract → verify spans → merge
                                                          │
                                                          ▼
                                                   [ result.json ]
```

### Inference backend: Ollama

* **The CLI manages the service for the one-shot flow.** A service already answering at
  the configured host is **adopted and never stopped** — it is not ours to take down; a
  one-line warning notes that its environment is unknown. Otherwise, when
  `service.manage` is true in `settings.json`, the CLI starts `ollama serve` with the
  tuning environment applied and **stops the whole process tree on exit** — Ctrl-C,
  Ctrl-Break and a closing console window included (exit 130). Stop only what we started,
  tracked explicitly, never inferred. `ollama serve` is an HTTP listener, not a model
  load, so this start is cheap.
* **Always `POST /api/chat`**, never the CLI (`ollama run`) as a subprocess.
* **Always send `keep_alive: 0`** so the model unloads when the run ends. This preserves
  the property the project has always had: a run leaves no VRAM held.
* **Always send `num_ctx` explicitly.** Ollama silently reduces an *automatic* context on
  OOM; an explicit one it cannot shrink ([tuning file §2.2](hardware-finetune.md)).
* **Structured output is the `format` field**, carrying `schemas/facts.json` verbatim.
  Ollama forwards it to llama.cpp as `json_schema`, so output is grammar-constrained.
  Never ask for JSON "by prompt" and never parse with a regex.
* Three settings are **service environment variables, not request options** —
  `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`. The VRAM budget
  depends on them. When the CLI starts the service it applies them from
  `settings.json`'s `service.env` block; an adopted service is trusted to have them and
  warned about once. See the tuning file §2.3.

### Configuration: `settings.json`

Generated at **build time** by `build.ps1`; **the binary only ever reads it.** No profiles,
no merging, no compiled-in fallback chain — one flat file describing one job.

```json
{
  "ollama": {
    "host": "http://127.0.0.1:11434", "model": "fact-extractor",
    "keep_alive": 0, "think": "max"
  },
  "service": {
    "manage": true,
    "command": "ollama",
    "startup_timeout_seconds": 60,
    "env": {
      "OLLAMA_FLASH_ATTENTION": "1",
      "OLLAMA_KV_CACHE_TYPE": "q8_0",
      "OLLAMA_NUM_PARALLEL": "1",
      "OLLAMA_MODELS": "D:\\Modelos\\Ollama"
    }
  },
  "source_model": "models/gemma-4-E2B-it-QAT-Q4_0.gguf",
  "chunk_tokens": 1000,
  "passes": 1,
  "options": {
    "num_ctx": 16384, "num_gpu": 99, "num_batch": 512,
    "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    "repeat_penalty": 1.0, "seed": 42
  }
}
```

`service.manage: false` restores the old behaviour — a service the user runs themselves,
and an error naming both fixes when none is reachable.

`source_model` records which `.gguf` the Ollama model was created from — provenance, not a
runtime path. Values originate in `hardware-finetune.md`; if the two disagree, the tuning
file is right.

**Keep the Modelfile free of `PARAMETER` lines.** They become defaults that request options
override, which is two sources of truth for one value.

**`--model` and `--no-think` override one field each, for the run only, never the file.**
`--model` replaces `cfg.Ollama.Model` in memory after `settings.Load` and before
`ollama.New`. Before the request's `think` field is ever set, `ollama.Client.SupportsThinking`
asks `/api/show` whether `cfg.Ollama.Model`'s capabilities include `thinking` — Ollama
returns `400 Bad Request` for `think` on a model that does not have that capability, rather
than ignoring it, so this checks first instead of assuming. Only when it does is `think` sent
explicitly — pinned to `settings.json`'s `ollama.think` by default (a bool or one of
`"low"|"medium"|"high"|"max"`; ships as `"max"`), `false` only when `--no-think` is passed —
because a reasoning model's own template already defaults to thinking-on
(`hardware-finetune.md` §1.8), so an unset field never actually turned it off. A model with no
thinking mode never sees the field at all, so `--no-think` is simply a no-op for it. Neither
flag writes `settings.json`, neither adds a second block of settings, and neither is read back
from disk — the boundary that keeps this from being the profile system §8 rules out.
`build.ps1`'s `$models` list can create more than one Ollama model (the default
`settings.json` names, plus any reachable only through `--model`), but `settings.json`'s shape
does not change: `source_model` still records provenance for the default model alone, and
`ollama.think` is one value shared by whichever model a run actually names.

**`$models[0]` is always the current benchmark winner.** Whichever model matches more gold
facts *in total* across the whole corpus (§5's scoreboard) is the one `settings.json` names
by default — that is the model a plain `fact-extractor.exe` run uses. A model that loses
that comparison is never deleted from `$models`: it drops to a later entry, still built,
still reachable with `--model`, kept specifically so it can be re-scored as a historical
baseline the next time `system-instruction.md` or the corpus changes — a regression there
could reorder the standings, and a baseline that no longer builds can't tell you that. This
is orthogonal to §5's tie-break rule: that one only resolves an equal score on a single
document for the scoreboard's bolding; this one picks the model every ordinary run actually
uses. As of §6, **Gemma 4 E2B (non-QAT)** backs the default `fact-extractor` name; **Gemma 4
E2B QAT** and **Qwen2.5-7B-Instruct-1M** are both historical baselines now, reachable with
`--model fact-extractor-gemma4-qat` and `--model fact-extractor-qwen` respectively.

**`passes` is a third global scalar, the same shape as `chunk_tokens`.** `passes: 1`
(the default) is today's pipeline, byte-for-byte. `passes > 1` follows each chunk's first
reply with one glean turn per extra pass, on the same message thread — `[chunk][pass 1
reply][glean turn][pass 2 reply]...` — so the model is asked to find what it missed rather
than repeat an identical request under greedy decoding, which would return the identical
list. The glean turn's wording is a **Go constant in `main.go` (`gleanInstruction`),
deliberately not folded into `system-instruction.md`**: that file must stay byte-identical on
every request to remain the cached prompt prefix (see Correctness, below), and the glean turn
only exists on the multi-pass path most runs never take. Each pass is verified against the
chunk and merged the same way chunks are (`facts.Merge`, deduping on the normalised
sentence), so a fact the model restates despite being told not to costs nothing. **Measured,
not just designed:** a `passes: 2` benchmark run against the current default model matched
the exact same 85/97 gold facts as `passes: 1`, on every one of the five cases, at 46% more
wall-clock time (584.8 s against 399.0 s) — the glean turn restated its own list rather than
finding anything new. `passes: 1` remains the shipped default on this evidence; the knob
stays because a different model or corpus may find the glean turn useful where this one did
not.

### Execution flow

```
1. Load settings.json. Fail clearly if absent or malformed.
2. Read system-instruction.md and prompt.md.
3. Ensure a service: adopt one already answering (leave it running on exit), else start
   `ollama serve` with the settings.json env and register the deferred stop + the
   Ctrl-C handler. Fail with an actionable message if neither is possible.
4. Check the model exists. Fail naming the model and how to create it.
5. Split the input into chunks of settings.chunk_tokens (character estimate — Ollama
   exposes no tokenizer). Refuse up front if system + largest chunk + headroom
   (scaled by settings.passes, see hardware-finetune.md section 2.6) > num_ctx.
6. For each chunk, for each of settings.passes: POST /api/chat { messages, format, options,
   keep_alive: 0 }. Pass 1 sends [system, chunk]; each pass after it appends the prior
   assistant reply and a glean-turn user message, then parses the new reply and verifies
   every verbatim span against that chunk.
7. Merge the per-chunk documents (across passes, then across chunks), drop duplicates,
   renumber ids.
8. Split the merged facts into cited and inferred, and locate every citation in the whole
   source (`internal/output`). This is presentation, after inference: it adds no claim and
   changes no fact.
9. Write result.json. Report facts, token counts, elapsed time and citation repairs.
10. Stop the service if — and only if — this run started it.
```

### Error handling (minimum required)

| Condition | Behaviour |
|---|---|
| `settings.json` missing or invalid | Name the file and the offending field. |
| No service and `service.manage` is false | Say so, and name both fixes: `ollama serve`, or `manage: true`. |
| `ollama serve` fails to start or never answers | Name the command tried and the timeout; suggest `service.command`. |
| Model not in `ollama list` | Name the model and how to create it. |
| Input too large for `num_ctx` | Refuse *before* sending, naming token counts and a concrete `chunk_tokens` value. |
| Reply hit the token cap | Save partial text to `result.raw.txt`, exit non-zero. |
| Malformed JSON despite the grammar | Save raw output to `result.raw.txt`, exit non-zero. |
| Ctrl-C | Stop a service we started (whole tree — stranded runners hold VRAM), exit 130. An adopted service is left alone. |

---

## 3. Hard constraints (non-negotiable)

- **6 GB VRAM** (RTX 3050 Laptop). Each 7B Q4_K_M is ~4.7 GB → **never two models loaded
  at once**. One run, one model.
- **Default context 16384**, sent explicitly ([tuning file §2.2](hardware-finetune.md)).
  Above the budget the driver silently spills to system RAM and halves throughput instead
  of failing, so OOM detection is never a substitute for the warning.
- **Verify GPU placement, do not assume it.** `ollama ps` must report `100% GPU`. Partial
  CPU offload is roughly half speed and produces no error
  ([tuning file §1.6](hardware-finetune.md)).
- Long documents are **chunked**, never handled by raising context. `chunk_tokens` is the
  only knob trading speed against missed facts ([tuning file §2.6](hardware-finetune.md)) —
  **raising the context does not license raising it.** A bigger window is room to *write*,
  not licence to feed the model a longer passage to *read*.

---

## 4. Correctness: what guarantees what

Three mechanisms, three jobs. Do not credit one with another's work.

| Property | Mechanism |
|---|---|
| Valid JSON, categories closed to the 7 allowed | The grammar (`format` → `json_schema`) |
| Stable output run to run | `temperature 0` + `top_k 1` — approximately |
| **Claims traceable to the source** | **`internal/facts.Verify`** |

- **Verify `verbatim`, do not trust it.** The CLI holds the source text, so `Verify` snaps
  every traceable span to the exact source characters and nulls the ones that are not
  there. After that pass `verbatim` is always `null` or a real substring.
  **Do not weaken this into a warning.**
- **`Verify` must not modify `confidence`.** It once forced dropped spans to `low`, which
  made a failed citation indistinguishable from an honest inference. Leaving the model's
  own value intact is what makes the output readable:

  | `verbatim` | `confidence` | Meaning |
  |---|---|---|
  | a span | any | Verified |
  | `null` | `low` | Inferred from the text |
  | `null` | `high` / `medium` | **A citation that failed verification** |

- **`position` is a report, not a guarantee.** `Verify` checks each span against *its own
  chunk*; `internal/output` locates it in the *whole* document. These can disagree in one
  narrow case: `textsplit` packs sentence units with a literal `

` the source may spell
  as a single space, so a span straddling that seam is a real substring of its chunk and not
  of the source. Normalisation collapses whitespace on both sides and resolves it; if a span
  still cannot be placed, the fact **keeps its citation** and reports `position: null`. A
  fact is never dropped or moved between arrays because a lookup failed — the arrays are
  partitioned on `verbatim` alone.
- **A position nothing checks is worse than none.** It reads as authoritative while sending
  the reader to the wrong line, so `internal/validate` re-derives every one independently
  and calls a mismatch a contract violation.
- Greedy decoding is **not** bit-reproducible, and does **not** prevent invented facts —
  greedy makes the argmax certain, not correct.
- The system instruction must be sent **first and byte-identical** on every request so it
  stays the cached common prefix. Any drift costs the whole prefix.

### The two error classes

- **17 facts where another run gave 18** — a near-tie fell the other way. Normal. Not a
  defect.
- **10 facts where 18 were present** — the model stopped extracting. A real and *silent*
  failure: schema-valid, fully verified, and missing a third of the document.

**Never write a test that asserts an exact fact count.** Assert traceability and a
plausible floor.

**A reasoning model adds a new way to reach the second class.** Thinking and content tokens
share one output budget with no separate accounting (`hardware-finetune.md` §1.8, §3 item
7) — a long enough trace can exhaust the budget before the fact array closes, which is the
same silent-truncation failure as above, reached by spending the budget on reasoning instead
of an oversized chunk. `res.Truncated()` (`done_reason: "length"`) still catches it the same
way; nothing about thinking being on by default bypasses that check.

---

## 5. Benchmark mode

`--benchmark` exists to answer one question: **did a new model, or an edited
`system-instruction.md`, make extraction worse?** It is not a general eval workbench.

**Pointing it at a new model:** add the `.gguf` and an entry in `build.ps1`'s `$models`
list, rebuild, then `fact-extractor.exe --model <name> --benchmark` (thinking is on by
default for a reasoning model; add `--no-think` to turn it off for the comparison). This is
exactly how the Gemma 4 comparison in `README.md` was produced —
`--model` and `--benchmark` compose freely, and `runBenchmark`'s banner names whichever
model `cfg.Ollama.Model` was resolved to, so the printed header always matches what was
actually scored.

Each corpus entry pairs a source text with gold facts anchored to source spans:

```json
{
  "source": "corpus/01-news.md",
  "gold": [
    { "verbatim": "Twelve of the 74 stations", "type": "numeric" }
  ]
}
```

- **Matching** resolves each gold span with `facts.Source.Find`. Because `Verify`
  guarantees every `verbatim` is `null` or a real substring, matching is character-exact —
  no fuzzy logic.
- **Scoring:** 0–3 unmatched gold facts → **PASS**. 4+ → **FAIL**.
- **Comparing two models' scores, a tie goes to the faster one.** Matched-fact count is the
  primary measure; when two models match the same number of gold facts on a document,
  generation throughput (`tok/s`, from the same run) breaks the tie in the faster model's
  favor. A benchmark exists to answer "did this get worse," and equal recall at a lower
  wall-clock cost is a strict improvement, not a wash — so the scoreboard never leaves a tie
  unresolved. This only ranks equal scores; it never substitutes for measuring recall itself.
- **Reported, not scored:** extracted count, spurious facts, `Verify` repair counts,
  contract validation (`internal/validate`, the same check `checkfacts` makes), type
  mismatches, and facts whose citation failed verification.
- **The validator is a second opinion by design.** `internal/validate` shares no code
  with `internal/facts` — if it did, it would confirm `verify.go`'s bugs instead of
  catching them. Keep them independent. A `verbatim: null` above `low` confidence is a
  *warning* there (a failed citation, the signal we deliberately preserve), never a
  contract violation.
- **Inferred facts (`verbatim: null`) are never graded** — they have no span to match.
- **Preflight:** confirm `ollama ps` reports full GPU placement. Benchmarking a
  CPU-offloaded model measures the wrong thing.
- **Every `--benchmark` run also writes `benchmark.json`** next to `result.json`
  (`internal/benchmark.Run`, encoded by `internal/benchmark.Encode`): the model, `think` value,
  `passes`, `chunk_tokens`, `num_ctx`, and every number `report()` prints to stderr, per case
  and totalled. This is the same reporting-only rule as everything else in this section — it
  does not feed `Score`, `Pass` or `Threshold`, and a write failure only logs a warning rather
  than failing the run. It exists so a matrix of models × `think` × `passes` can be compared
  without transcribing terminal output by hand, which is how every number in `README.md`'s
  scoreboard was produced before it existed.

### The claims-about-code case

`04-code-claims` is a mixed document: prose and comments making claims, with the code that
implements them alongside. Its gold list is **eight planted pairs** — a claim and its
evidence, four agreeing and four contradicting — so a pair scores 2 when both halves are
extracted and 1 when only one is.

It measures **surfacing, not detecting**: whether both halves reach the reader with verified
citations. The tool never says two facts contradict each other; that is the reader's
judgement, and moving that line would make this a fact checker, which §1 says it is not.

Two constraints when editing that corpus: **both halves of a pair must be citable spans** —
a fact about an *absence* has no span, becomes `verbatim: null`, and is not graded — and the
document must **fit in one chunk**, because a pair split across chunks is invisible to the
model by construction.

**This case is a capability probe, not a gate.** The current model scores **8/16**: it reads
prose claims and skims code blocks longer than a couple of lines — every evidence span it
found sits in a two-line block, every one it missed in a block of seven or more. That number
is the measurement. It is expected to stay red until a model does better, and a
reasoning-capable model is the obvious next thing to point at it. **Never trim the gold list
to make it pass.**

### The multi-chunk density case

`05-survey` is a synthetic literature-review document — invented authors, institutions and
findings in the corpus's existing fictional world, not summarised from any real source. It
probes the shape `chunk_tokens` tuning exists to worry about: attribution-dense prose, a
tab-separated table pasted as plain text, and one paragraph long enough (over 1,000 estimated
tokens) to force `textsplit`'s sentence-level fallback rather than staying a single paragraph
unit. At `chunk_tokens = 1000` it splits into three chunks, the only corpus document that
does.

**This case is also a capability probe, not a gate**, and it is expected to be the hardest one
in the suite. The current model scores **13/23**: it misses both gold spans drawn from the
table and most of the bibliography-style source list, the same skimming pattern
`04-code-claims` shows against code — dense, list-shaped, non-narrative text is where recall
drops hardest. It also produced far more failed citations here (21) than on any other case,
consistent with dense citation-heavy prose being harder to quote back exactly. That number is
the measurement, not a bug in the corpus. **Never trim the gold list to make it pass.**

---

## 6. Where this stands, and what's next

Three models have been scored so far, same corpus, same `system-instruction.md`,
`chunk_tokens = 1000`: **76%** (97 gold facts, 74 matched) against
`Qwen2.5-7B-Instruct-1M` Q4_K_M; **86%** (83 matched) against `Gemma 4 E2B QAT`, thinking on
by default; and **88%** (85 matched) against `Gemma 4 E2B` (non-QAT), also thinking on by
default. Non-QAT Gemma wins outright over both — a clean win over QAT on `05-survey`
(18/23 against 16/23) and an exact tie everywhere else, so per this file's rule it now backs
the default `fact-extractor` name; QAT and Qwen are both historical baselines, reachable
with `--model fact-extractor-gemma4-qat` and `--model fact-extractor-qwen`. Two cases fail by
design under both Gemma builds today — `04-code-claims`, `05-survey` — down from three under
Qwen (`02-research` now scores a clean 18/18 on either Gemma). **Do not raise
`benchmark.Threshold` or trim a gold list to make any of them pass**; the full per-document
account for all three models lives in [`README.md`](README.md#the-experiment).

The remaining failures share one shape, diagnostically, under all three models: the model
stops early on text that is not narrative prose — code blocks, a plain-text table, a
bibliography-style source list, subordinate clauses — and hands back a list that is
schema-valid but incomplete. Nothing in §4 catches that class of failure on its own: the
grammar guarantees shape, never exhaustiveness, and no sampling parameter in §2.4 of the
tuning file touches it either.

**A reasoning-capable model was the first of two directions this section named as more
promising than further tuning — it has now been tried, not just predicted.**
`hardware-finetune.md` §1.8 has the full measurement: Gemma's thinking budget appears to buy
something closer to a coverage sweep before the array closes, most visibly on
`02-research`'s qualifiers and appositives coming out as separate facts instead of folding
into a headline claim. The honest cost predicted in §3 item 7 landed too — thinking and
content tokens share one output budget with no separate accounting, confirmed directly
(~2.4× the generation tokens for one chunk) — though on a smaller, faster model the
wall-clock cost is less severe than that prediction worried, even after paying it. What
remains open: whether this generalizes past one small model, and whether a larger reasoning
model closes the two remaining probes rather than only narrowing them.

**A same-architecture follow-up landed by accident, not by design.** `hardware-finetune.md`
§1.9 compares the QAT release above against the non-QAT Gemma 4 E2B release quantized the
ordinary way: same architecture, same thinking mechanism, genuinely different weights (QAT
is fine-tuned for quantization robustness, not just a different quant scheme of the same
checkpoint). It won on every axis measured — higher score, faster generation, shorter
thinking traces — which is why it, not QAT, backs the default name today.

**Two more levers were tried against the same "stopped early" failure shape, and both came
back negative on this model and corpus — but one is kept as the shipped default anyway.**
Ollama's `think` field accepts `"low" | "medium" | "high" | "max"` as well as a boolean, and
Gemma 4 E2B's own capabilities list `thinking`, so a direct probe sent five identical requests
against `01-news`, one per value. All five — including plain `true` — returned byte-identical
`eval_count` (2454), thinking length (3752 characters) and content, while `think: false` on
the same request measured 1441 tokens and no thinking trace at all. The model accepts a level
string without a `400`, but this architecture's Ollama template does not act on it: it is a
no-op, not a dial, on this model. **`ollama.think` ships as `"max"` regardless** — a
deliberate bet, not an oversight: it costs nothing on a model that ignores it, and pays off
automatically on a future thinking-capable model whose template actually varies by level,
without a settings.json edit when that model arrives. `internal/ollama.Client.Chat`'s `Think`
field was widened from `*bool` to a bool-or-string type to carry it; `SupportsThinking` still
gates the whole thing exactly as before, so a model with no thinking capability at all never
sees the field. Second, the `passes` scalar described under Configuration above: a `passes: 2`
benchmark run matched the exact same 85/97 gold facts, case for case, as `passes: 1`, at 46%
more wall-clock time — the glean turn restated its own list rather than surfacing anything
new. `passes: 1` stays the default here on that evidence; both `passes` and the wider `think`
type stay available in the code for a model or corpus where either might behave differently.

A second direction remains untried:

- **A model trained on claim-plus-evidence-span data**, not merely fine-tuned for chat.
  FEVER's evidence-selection stage — pick the sentence supporting a claim — is the same
  operation as citing a `verbatim` span and targets this project's actual recall problem;
  its claim-*verification* stage would turn this into a fact checker, which §1 says this
  project deliberately is not.

This direction is not trainable on the hardware in `hardware-finetune.md` §1.4 — 6 GB will
not hold a fine-tuning run over a 7B-class model, so it would mean a rented GPU, unlike the
reasoning-model comparison above, which ran entirely on this machine. The corpus
(`corpus/*.gold.json`, 97 hand-written facts) stays a held-out benchmark either way: it is
sized to score a model, not to train one, and folding it into training data for whatever
comes next would make every number in §5 meaningless.

---

## 7. Project conventions

- **The schema and the system instruction must stay in sync.** `schemas/facts.json` is the
  grammar's contract; `system-instruction.md` is the prose that teaches the model what each
  field and enum value *means*. They can drift silently — a field or category added to one
  and not explained in the other produces output the grammar allows and the model was never
  taught. `main_test.go` (`TestSystemInstructionMatchesSchema`,
  `TestFactTypeVocabularyMatchesSchema`) checks both directions against the embedded
  copies on every `go test`. Editing either file without re-running the tests is how this
  drifts.
- **I/O files:** `system-instruction.md` (prompt), `prompt.md` (input), `result.json`
  (output, shaped by `internal/output` — see §1), `result.raw.txt` (rescued output on
  failure), `settings.json` (configuration), `benchmark.json` (per-`--benchmark`-run report,
  written by `internal/benchmark.Encode` — see §5). Do not rename these.
- The schema and the system instruction are `go:embed`-ed as fallbacks; a file beside the
  exe wins.
- **`settings.json` is generated by the build, never by the binary.**
- Everything in this repository — code, identifiers, comments, docs, prompts, corpus
  inputs, commit messages — in **plain English**.
- Standard Go: `gofmt`, errors wrapped with `fmt.Errorf("...: %w", err)`, **stdlib only**
  (no external dependencies).
- Static build: `go build -ldflags="-s -w" -o fact-extractor.exe`.

---

## 8. Do not

- Add runtime dependencies on LM Studio, Python, or a bundled `llama-server.exe`. The
  engine is Ollama, reached over HTTP.
- Stop an Ollama service this run did not start — adopted services are left running.
- Let a service we started outlive the run, or kill only its parent process: stranded
  runner subprocesses hold ~4.7 GB of VRAM. Take down the tree.
- Ask for JSON "by prompt" and parse with a regex — always use `format`.
- Write `settings.json` from the binary.
- Put `PARAMETER` lines in the Modelfile.
- Let `Verify` overwrite `confidence`.
- Renumber ids when splitting the output into arrays, or partition those arrays on anything
  other than whether `verbatim` is null.
- Report a `position` that nothing independently checked.
- Add flags beyond `--benchmark`, `--model` and `--no-think`. Each overrides exactly one
  `settings.json` field for the run and nothing else — a fourth flag needs the same
  justification these two got, not a default yes.
- Reintroduce profiles (a `models` block, an alias map, per-model `options`) or a non-JSON
  output mode. `--model` names an Ollama model directly; it is not a profile system, and
  `settings.json`'s shape does not change to support it — see §2's Configuration section.
- Make `think` or `passes` vary per model, or move either into a `models` block. Both are
  global scalars in `settings.json`, the same shape as `chunk_tokens` — a recall dial the
  build owns, not a per-model setting; that would be the profile system this file already
  rules out, just spelled a different way.
- Fold the glean-turn wording (`gleanInstruction` in `main.go`) into `system-instruction.md`.
  That file must stay byte-identical on every request to remain the cached prompt prefix (§4);
  the glean turn only exists on the `passes > 1` path most runs never take.
- Assert an exact fact count in a test.
- Commit `.gguf` files or binaries to git.

## 9. Primary directives

- YOU SHALL NEVER COMMIT TO GIT
