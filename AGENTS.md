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
fact-extractor.exe                      # system-instruction.md + prompt.md -> result.json
fact-extractor.exe --benchmark          # score the gold corpus, exit 0 or 1
fact-extractor.exe --benchmark --coder  # score the same corpus under the second model
```

It does exactly one job. There is no profile system and no output format other than JSON.
Model, context and sampling come from `settings.json`; input and output filenames are
fixed.

There are exactly **two flags**: `--benchmark`, and `--coder`, which selects
`ollama.coder_model` **for measurement only**. The bound on that second model is a rule,
not a preference — see §7.

### The identity to protect

**A fact extractor whose citations are traceable to the source.** Every `verbatim` span in
the output is either a real substring of the input or `null` — never something the model
merely believes it copied. That guarantee, not the model and not the runtime, is the
product.

It is **not an agent**: no loop, no tool use, no autonomy. Control flow is fixed before the
model is ever called.

### Output schema (frozen)

`schemas/facts.json` — `id`, `fact`, `type`, `confidence`, `verbatim`. Seven categories:
`quote`, `numeric`, `event`, `entity`, `definition`, `causal`, `other`.

The schema is a theory of what a fact is. Changing it invalidates every comparison between
models and between system instructions, so it does not change casually.

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
  OOM; an explicit one it cannot shrink ([tuning file §4](hardware-finetune.md)).
* **Structured output is the `format` field**, carrying `schemas/facts.json` verbatim.
  Ollama forwards it to llama.cpp as `json_schema`, so output is grammar-constrained.
  Never ask for JSON "by prompt" and never parse with a regex.
* Three settings are **service environment variables, not request options** —
  `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`. The VRAM budget
  depends on them. When the CLI starts the service it applies them from
  `settings.json`'s `service.env` block; an adopted service is trusted to have them and
  warned about once. See the tuning file §4.

### Configuration: `settings.json`

Generated at **build time** by `build.ps1`; **the binary only ever reads it.** No profiles,
no merging, no compiled-in fallback chain — one flat file describing one job.

```json
{
  "ollama": { "host": "http://127.0.0.1:11434", "model": "fact-extractor", "keep_alive": 0 },
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
  "source_model": "models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf",
  "chunk_tokens": 3000,
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

### Execution flow

```
1. Load settings.json. Fail clearly if absent or malformed.
2. Read system-instruction.md and prompt.md.
3. Ensure a service: adopt one already answering (leave it running on exit), else start
   `ollama serve` with the settings.json env and register the deferred stop + the
   Ctrl-C handler. Fail with an actionable message if neither is possible.
4. Check the model exists. Fail naming the model and how to create it.
5. Split the input into chunks of settings.chunk_tokens (character estimate — Ollama
   exposes no tokenizer). Refuse up front if system + largest chunk + headroom > num_ctx.
6. For each chunk: POST /api/chat { messages, format, options, keep_alive: 0 }.
   Parse the reply, then verify every verbatim span against that chunk.
7. Merge the per-chunk documents, drop duplicates, renumber ids.
8. Write result.json. Report facts, token counts, elapsed time and citation repairs.
9. Stop the service if — and only if — this run started it.
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
- **Default context 16384**, sent explicitly ([tuning file §3](hardware-finetune.md)).
  Above the budget the driver silently spills to system RAM and halves throughput instead
  of failing, so OOM detection is never a substitute for the warning.
- **Verify GPU placement, do not assume it.** `ollama ps` must report `100% GPU`. Partial
  CPU offload is roughly half speed and produces no error
  ([tuning file §4.1](hardware-finetune.md)).
- Long documents are **chunked**, never handled by raising context. `chunk_tokens` is the
  only knob trading speed against missed facts ([tuning file §7](hardware-finetune.md)) —
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

---

## 5. Benchmark mode

`--benchmark` exists to answer one question: **did a new model, or an edited
`system-instruction.md`, make extraction worse?** It is not a general eval workbench.

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
- **Reported, not scored:** extracted count, spurious facts, `Verify` repair counts,
  contract validation (`internal/validate`, the same check `checkfacts` makes), type
  mismatches, and facts whose citation failed verification.
- **The validator is a second opinion by design.** `internal/validate` shares no code
  with `internal/facts` — if it did, it would confirm `verify.go`'s bugs instead of
  catching them. Keep them independent. A `verbatim: null` above `low` confidence is a
  *warning* there (a failed citation, the signal we deliberately preserve), never a
  contract violation.
- **Inferred facts (`verbatim: null`) are never graded** — they have no span to match.

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
- **Preflight:** confirm `ollama ps` reports full GPU placement. Benchmarking a
  CPU-offloaded model measures the wrong thing.

---

## 6. Project conventions

- **I/O files:** `system-instruction.md` (prompt), `prompt.md` (input), `result.json`
  (output), `result.raw.txt` (rescued output on failure), `settings.json` (configuration).
  Do not rename these.
- The schema and the system instruction are `go:embed`-ed as fallbacks; a file beside the
  exe wins.
- **`settings.json` is generated by the build, never by the binary.**
- Everything in this repository — code, identifiers, comments, docs, prompts, corpus
  inputs, commit messages — in **plain English**.
- Standard Go: `gofmt`, errors wrapped with `fmt.Errorf("...: %w", err)`, **stdlib only**
  (no external dependencies).
- Static build: `go build -ldflags="-s -w" -o fact-extractor.exe`.

---

## 7. Do not

- Add runtime dependencies on LM Studio, Python, or a bundled `llama-server.exe`. The
  engine is Ollama, reached over HTTP.
- Stop an Ollama service this run did not start — adopted services are left running.
- Let a service we started outlive the run, or kill only its parent process: stranded
  runner subprocesses hold ~4.7 GB of VRAM. Take down the tree.
- Ask for JSON "by prompt" and parse with a regex — always use `format`.
- Write `settings.json` from the binary.
- Put `PARAMETER` lines in the Modelfile.
- Let `Verify` overwrite `confidence`.
- Add a third flag. The CLI takes `--benchmark` and `--coder`, and nothing else.
- Grow the second model into a profile system. It exists **for A/B measurement only**: one
  flag, one `ollama.coder_model` field, no instruction fork, no profile map. A *third*
  model, or a flag that bundles a different instruction with a different model, is the
  profile system this project deleted. The bound is the whole reason the rule was relaxed.
- Reintroduce profiles or a non-JSON output mode.
- Assert an exact fact count in a test.
- Commit `.gguf` files or binaries to git.
