# Hardware Fine-Tuning: Qwen2.5-7B-Instruct-1M on an RTX 3050 6 GB

This file owns **every number in this project that exists because of this specific
machine**: VRAM budgets, context limits, runtime settings and sampling values.
[`AGENTS.md`](AGENTS.md) references this file rather than restating those numbers, so
there is one place to change when the hardware, the driver or the runtime moves.

The split is: *`AGENTS.md` says what the tool is and does. This file says what numbers to
run it with on this GPU.*

| | |
|---|---|
| Validated | 2026-08-23 |
| GPU driver | 591.66 |
| Runtime | **Ollama** 0.32.15 (llama.cpp underneath) |
| Baseline measurements | llama.cpp **b10588** (`70adb1b4c`), direct — see §3 |
| Model | `Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` (~4.68 GB on disk) |

Every figure below is reproducible with the command printed beside it. Nothing here is an
estimate unless it is labelled as one. Section 10 lists the commands to regenerate the
whole file after a driver, build or model change.

---

## 1. Measured hardware

```
$ llmfit system
CPU: AMD Ryzen 5 7235HS (8 cores)
Total RAM: 15.69 GB
Available RAM: 10.72 GB
RAM Bandwidth: ~38 GB/s (measured)
Backend: CUDA
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.00 GB VRAM, CUDA)
```

```
$ nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, 187 MiB, 591.66
```

**The hard constraint that drives everything else:** 6144 MiB of VRAM against a 4.68 GB
model file. One 7B model fits at a time, with roughly 1 GB left over for the KV cache and
the compute buffers. There is no configuration in which two models are loaded together.

The desktop itself occupies part of the card — observed between **187 MiB** (nothing
running) and **696 MiB** (normal desktop session). Budget against the higher figure.

---

## 2. Model fit — and why the previous numbers were wrong

```
$ llmfit info "Qwen/Qwen2.5-7B-Instruct-1M"
Overall Score: 82.0 / 100
  Quality 76   Speed 91   Fit 63   Context 100
Runtime: llama.cpp (baseline est. ~36.5 tok/s)
Fit Analysis:
  Status: Marginal
  Run Mode: GPU
  Memory Utilization: 89.2% (5.4 / 6.0 GB)
Estimate Basis:
  Method: per-backend heuristic constant - GPU not in the bandwidth table,
          so expect a wide error band.
  Baseline error band is roughly +/-30%.
```

The previous version of this document reported **90.4 / 100**, **Fit: Good**, **79.1%
utilisation** and **38.1 tok/s**. Those numbers are real, but they belong to a different
row of `llmfit fit`: `Qwen2.5-Coder-7B-Instruct-GPTQ-Int4` and its AWQ twin, scored **on
vLLM**. Different model, different quantisation, and a runtime this project does not use.

The model actually in use scores **82.0** and rates **Marginal**.

### Why "Marginal" overstates the problem

llmfit says plainly that this GPU is absent from its bandwidth table. Measured against
`nvidia-smi`, its VRAM projections run consistently **~600 MB high**:

| Context | llmfit estimate (q8_0 KV) | Measured `llama-server` | llmfit error |
|---|---|---|---|
| 8192 | 5.14 GB | 4645 MiB (4.54 GB) | +0.60 GB |
| 16384 | 5.35 GB | 4891 MiB (4.78 GB) | +0.57 GB |
| 32768 | 5.79 GB | 5208 MiB (5.09 GB) | +0.70 GB |

Reproduce with `llmfit plan "Qwen/Qwen2.5-7B-Instruct-1M" --context N --kv-quant q8_0`.

**Direct measurement wins over the heuristic.** The "Marginal / 89.2%" verdict is computed
from the inflated estimate. In practice 16384 leaves 1253 MiB unallocated on the card
before the desktop's share, which is roughly **0.5-1.0 GB of real headroom** depending on
what else is displaying.

Treat llmfit as corroboration and a sanity check, not as the source of truth for this
machine. It is authoritative on model *ranking*; it is guessing at this GPU's *capacity*.

---

## 3. VRAM and context budget

Weights at Q4_K_M are about 4.7 GB. KV cache measures **~30 KB/token** with `q8_0` across
the 4k-16k range.

Measured with `nvidia-smi` while `llama-server` held the model, desktop idling at 696 MiB:

| Configuration | `llama-server` | Total incl. desktop | Verdict |
|---|:---:|:---:|:---|
| `-c 4096` | 4523 MiB | 5219 MiB | fits |
| `-c 8192` | 4645 MiB | 5341 MiB | fits |
| **`-c 16384`** | **4891 MiB** | **5587 MiB** | **default - comfortable** |
| `-c 32768` | 5208 MiB | 5904 MiB | practical ceiling, ~0.24 GB spare |
| `-c 65536` | 5176 MiB | 5872 MiB | **spilled to system RAM** |

> **These are `llama-server` baseline figures**, taken by driving the binary directly.
> Ollama adds its own process overhead and its own memory estimator (§4), so the real
> footprint under Ollama will differ somewhat. The table remains the reference for *how
> context scales* — ~30 KB/token, 246 MiB from 8192 to 16384 — and the re-measurement
> commands are in §10. Confirm placement with `ollama ps` (§4.1), which is the check that
> actually matters.

### The default is 16384

Going from 8192 to 16384 costs **246 MiB** and **no measurable speed**. Prefill and
generation scale with the tokens actually used, not with the window reserved; unused
context is idle VRAM, not wasted compute. There is no performance argument for staying at
8192 - it is a smaller room for the same work.

What the extra room is *for* is covered in section 7. It is output headroom, not licence
to feed the model longer passages.

**16384 is safe for both installed models.** `--coder` selects
`Qwen2.5-Coder-7B-Instruct-Q4_K_M`, whose trained context is **32768** (`ollama show
fact-extractor-coder`) against the main model's 1010000. Both are comfortably above the
window, so `num_ctx` stays one number rather than becoming model-specific. Check this
again before installing any third model — a model trained below 16384 would silently
degrade rather than refuse.

### The failure mode above the budget is silence, not a crash

On this WDDM driver, exceeding VRAM does not abort. The allocation spills into system RAM
and throughput collapses with **no error of any kind in the server log**. Measured at
`-c 65536`:

| | Within budget | Spilled |
|---|---|---|
| Generation | 31.4 tok/s | **16.2 tok/s** |
| Prefill | ~1200 tok/s | **272 tok/s** |

A silent halving of speed is harder to notice than a crash. This is why the CLI prints a
budget warning above 16384 rather than relying on OOM detection, and why large `--ctx`
values are warned about but never blocked - a smaller custom model may afford more.

**If an allocation failure is ever reported: reduce context first, never `-ngl`.** Moving
layers to the CPU costs more speed than halving the context.

---

## 4. Runtime settings under Ollama

Ollama runs llama.cpp underneath, so the tuning carries over — but it is **split across
two places**, and that split is the single most important thing in this section.

| llama-server flag | Ollama equivalent | Set where |
|---|---|---|
| `-c 16384` | `num_ctx: 16384` | `settings.json` → `options` |
| `-ngl 99` | `num_gpu: 99` | `settings.json` → `options` |
| `-b 512` `-ub 512` | `num_batch: 512` | `settings.json` → `options` |
| `-fa on` | `OLLAMA_FLASH_ATTENTION=1` | **service environment** |
| `-ctk q8_0` `-ctv q8_0` | `OLLAMA_KV_CACHE_TYPE=q8_0` | **service environment** |
| `--parallel 1` | `OLLAMA_NUM_PARALLEL=1` | **service environment** |
| `--cache-reuse 256` | not exposed | irrelevant — inert in this workload, see below |

**The bottom three are service environment, not request options** — fixed when the
service starts. They matter because **the measured 4891 MiB at 16384 depends on `q8_0`
being active**: at fp16 the KV cache doubles (~30 KB/token becomes ~56) and the budget in
§3 no longer holds.

**When the CLI starts the service itself** (`service.manage: true` in `settings.json`,
the shipped default) it applies them from the `service.env` block — `settings.json` owns
the whole tuning story and nothing below is needed.

**When Ollama is run independently** — `manage: false`, or a service that was already up
and got adopted — the variables must be set for that service by hand, and the CLI can
only warn. On Windows, set and restart it:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_NUM_PARALLEL 1
```

### Always send `num_ctx` explicitly

Ollama's scheduler reduces the context by itself after a CUDA OOM and retries
(`reduceAutoNumCtxForLoadOOM`, `server/sched.go`) — **but only when the context is
"auto"**. An explicitly supplied `num_ctx` cannot be silently shrunk.

Silent degradation is the failure mode this whole document exists to fight, so
`settings.json` always sends an explicit `num_ctx`. Never leave it to the default.

### Ollama's own VRAM estimate assumes f16, and is therefore wrong here

`PredictServerVRAM` in `llm/llama_server.go` computes the KV cache as:

```go
// KV cache: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (f16)
kvCache := 2 * layers * kvHeads * headDim * uint64(numCtx) * 2
```

The `2 bytes` is **hardcoded f16**, even when `OLLAMA_KV_CACHE_TYPE=q8_0` halves the real
cost to 1 byte per element. Ollama therefore over-estimates this model's footprint and may
decide to leave layers on the CPU that would have fitted on the GPU.

This is the same class of failure as the WDDM spill in §3 — a silent halving of speed
rather than an error — reached by a different route. §4.1 is how you catch it.

### 4.1 `ollama ps` detects the failure that used to be invisible

The reason §3 warns instead of trusting OOM detection is that partial CPU offload was
undetectable on the old stack: throughput simply halved, with nothing in any log.

**Ollama reports the split directly.** This is a genuine capability the previous runtime
did not offer:

```powershell
ollama ps
```

```
NAME              ID              SIZE      PROCESSOR    UNTIL
fact-extractor    a1b2c3d4e5f6    5.2 GB    100% GPU     4 seconds from now
```

`100% GPU` is the required state. Anything of the form `30%/70% CPU/GPU` means layers were
offloaded and the run is measuring the wrong thing. **Check this after any tuning change,
and before trusting any benchmark result.**

### 4.2 Ollama never lowers GPU layers on OOM

This project arrived by measurement at the rule *reduce context first, never `-ngl`*.
Ollama's scheduler already implements exactly that: on OOM it reduces context (when auto)
and evicts other models, and **neither path lowers the user-specified GPU layer count**
(`server/sched.go`).

The rule is now enforced by the runtime rather than merely advised here.

### `--cache-reuse` was blamed for something it does not do

Recorded because the mistake is easy to repeat. It was previously held responsible for
run-to-run variance. Checked against llama.cpp b10588, that is wrong — the binary exposes
two independent mechanisms:

```
--cache-prompt, --no-cache-prompt   whether to enable prompt caching (default: enabled)
--cache-reuse N                     min chunk size to attempt reusing from the cache
                                    via KV shifting, requires prompt caching to be enabled
```

In `tools/server/server-context.cpp`, `get_common_prefix()` runs whenever `cache_prompt`
is set, and the `cache_reuse` loop then starts at `head_c = n_past` — **only past the
point where the prefix diverges.**

Every request this tool sends is `[system instruction][chunk N]`. The system instruction
*is* the common prefix, so it is reused by ordinary prompt caching with no KV shifting
involved. For `--cache-reuse` to fire at all, two different chunks would have to share
hundreds of consecutive identical tokens, which prose does not do.

Ollama does not expose the flag, and nothing is lost by that. Prompt-prefix caching still
happens; it is what actually matters for chunked documents.

---

## 5. Sampling

Sent **per request**, in Ollama's `options` block. Never baked into the Modelfile — see
the note at the end of this section.

| Parameter | Value | What it actually buys |
|---|:---:|---|
| `temperature` | `0.0` | Greedy decoding - the highest-probability token every step. Buys **stability**, not correctness. See section 6. |
| `top_k` | `1` | Greedy. Redundant with `temperature 0` but explicit. |
| `top_p` | `1.0` | **A no-op.** Under greedy decoding the argmax is taken regardless of the nucleus. Documented only so nobody mistakes it for a safeguard. |
| `repeat_penalty` | `1.0` | Disabled, correctly. Not because it would "corrupt JSON braces" - the grammar makes that structurally impossible - but because a fact list legitimately repeats tokens (field names, recurring entities) and penalising them skews extraction. |
| `seed` | `42` | Pins the sampler. Does **not** make runs bit-identical; see section 6. |

Structured output is not an `options` entry. It is Ollama's top-level **`format`** field,
which accepts a full JSON Schema — `schemas/facts.json` is sent verbatim. Ollama forwards
it to llama.cpp as `json_schema`, so the output is GBNF-constrained exactly as before.
This is what guarantees parseable output.

**No sampling parameter on this list prevents hallucination.** That claim was the previous
document's central error and it is worth stating flatly.

> **Keep the Modelfile free of `PARAMETER` lines.** `ollama create` accepts them, and they
> become defaults that request options then override — two sources of truth for one value,
> which is exactly the drift this file exists to prevent. `settings.json` owns the
> parameters; the Modelfile owns only `FROM <gguf>`.

---

## 6. Where correctness actually comes from

Three separate mechanisms do three separate jobs. The previous document credited all three
to sampling.

| Property | Mechanism |
|---|---|
| Output is valid JSON, categories closed to the 7 allowed | **GBNF grammar** (`response_format: json_schema`) |
| Output is stable run to run | `temperature 0` + `top_k 1` - approximately |
| Claims are traceable to the source | **`internal/facts/verify.go`** |

### Greedy decoding does not prevent invention

`temperature 0.0` makes the model pick its *most likely* token, with total confidence. If
the most likely continuation is wrong, greedy picks it anyway - and picks it every time.
Recorded example: the model emitted "**Eleven** of the 74 stations" where the source reads
"**Twelve**", at temperature 0.0.

What catches that is the verification pass. The CLI holds the source text, so it checks
every `verbatim` span against it: a traceable span is snapped to the exact source
characters, and a span that is not present becomes `null`. After that pass, `verbatim` is
by construction either `null` or a real substring of the input.

**The verifier does not touch `confidence`.** It used to force dropped spans to `low`,
which made a failed citation indistinguishable from an honest inference. Leaving the
model's own assessment intact makes the output readable:

| `verbatim` | `confidence` | Meaning |
|---|---|---|
| a span | any | Verified — a real substring of the source |
| `null` | `low` | Inferred from the text, never claimed to be a quote |
| `null` | `high` / `medium` | **A citation that failed verification** |

That third row is the number worth watching when comparing two models: it counts how often
a model fabricates a span outright.

Measured on the long-report demo: **62 exact, 2 snapped, 3 dropped** - after which the
independent `checkfacts` validator graded all 64 remaining spans exact.

**This is the tool's actual safety guarantee.** It holds on every run regardless of what
the sampler did, and it is a stronger promise than determinism would have been.

### The two error classes

Output varies slightly between identical runs - two runs over the same input produced 17
and 18 facts. This is normal for batched GPU inference (floating-point reduction order is
not fixed, and the amount of cached prefix changes batch composition) and it is **not
worth engineering against**:

- **17 facts instead of 18** - one marginal claim landed on the other side of a near-tie.
  Both sets are fully verified. Not a defect.
- **10 facts instead of 18** - the model *stopped extracting*. A real and **silent**
  failure: the output is schema-valid, fully verified, and omits a third of the document.

Only the second is a defect. Section 7 is about preventing it.

> Never write a test that asserts an exact fact count. Assert traceability and a
> plausible floor instead.

---

## 7. The recall dial

Causes of under-extraction, ranked by how much they bite on this setup:

1. **Chunk size** - dominant. Recall falls as the passage lengthens; attention dilutes and
   mid-chunk facts get skipped. **`chunk_tokens` is the only knob that genuinely trades
   speed against mistakes.**
2. **Output headroom** - a fact list can be as long as its source. The CLI catches a hard
   `max_tokens` truncation, but a model that simply *finishes early* is not truncation and
   nothing flags it.
3. **The grammar permits stopping** - `json_schema` guarantees a valid document, never a
   complete one. `]` is a legal token after any finished fact object. The grammar enforces
   shape; it cannot enforce exhaustiveness.
4. **The system instruction** - "extract EVERY fact" carries real weight, but it is the
   weakest of the four.

Note what is *absent*: `temperature`, `top_p` and `repeat_penalty` have no effect on
recall at all.

### Raise context, not chunk size

This is the rule that follows, and it is the one the previous document got backwards. It
claimed 16384 was needed to "ingest long articles" - long articles are **chunked**, not
ingested.

At `num_ctx 8192` with 3000-token chunks the budget is roughly 1,700 (system instruction) +
3,000 (chunk) + reply + 512, which boxes in the fact list. At `num_ctx 16384` with **chunks
held at 3000 or lowered**, the entire gain goes to letting the model emit as many facts as
the passage actually contains - attacking cause 2 directly.

**16384 buys the model room to *write* a long fact list. It must not buy it a longer
passage to *read*.** The two rising together is what silently costs recall.

### Moving the dial

Default is **3000**, the measured working value.

| Direction | Effect |
|---|---|
| **Lower** (1500-2500) | Higher recall, more requests, slower on long documents. Choose this when a missed fact matters more than elapsed time. |
| **Default** (3000) | Balanced; validated by the corpus suite. |
| **Higher** (4000+) | Fewer requests and faster, at rising risk of silent omission from dense passages. |

Chunk boundaries follow paragraphs, then sentences, then a hard split.

> **Chunk sizing is now an estimate, not a measurement.** The old stack sized chunks with
> the model's own tokenizer via `POST /tokenize`. Ollama does not expose tokenization on
> its public API — `/tokenize` exists only on the llama-server subprocess it manages
> internally — so `textsplit.Estimate` (a character-based approximation) is the only
> option. This is why chunks are sized conservatively against a 16384 window: the slack
> absorbs the estimate's error. Real token counts still come back in each response as
> `prompt_eval_count`, which is enough for reporting but arrives too late for sizing.

---

## 8. Speed levers, ranked

Generation on this card is memory-bandwidth-bound at ~38 GB/s. **Sampling parameters cost
nothing in speed**, so there is no balance to strike there. The real knobs:

1. **`chunk_tokens`** - the only true speed/quality tradeoff. See section 7.
2. **`num_batch` 512 to 1024** - trades a little VRAM for prefill throughput. Untested on
   this machine; measure before adopting.
3. **`num_ctx` 16384** - ~246 MiB, no speed cost. Already the default.
4. **KV `q4_0`** (`OLLAMA_KV_CACHE_TYPE=q4_0`) - **rejected.** `llmfit` shows it would free
   another 50% (5.14 GB at 16k), enough to reach 32768. It degrades precisely the citation
   fidelity this tool exists to provide. Not a trade worth making.
5. **Full GPU placement** - not a dial but a precondition. A partially offloaded model is
   roughly half speed; `ollama ps` (§4.1) is how you know.

Measured throughput for reference: generation **25-32 tok/s**, prefill **1100-1400 tok/s**,
model load **3.3-5.8 s** from NVMe (about 50 s from USB 3.0).

Note that llmfit's 36.5 tok/s is a heuristic with a stated +/-30% band, and the real figure
sits at the low end of it. Prefer the measured range.

---

## 9. Changed from `hardware-recommendations.md`

This file replaced `hardware-recommendations.md`. Five corrections were made, and they all
still stand:

1. **Model scores re-attributed.** 90.4 / Good / 79.1% / 38.1 tok/s belonged to
   `Qwen2.5-Coder-7B-Instruct-GPTQ-Int4` on vLLM. The model in use scores 82.0 / Marginal
   / 89.2%.
2. **Measurement promoted over estimate.** llmfit runs ~600 MB pessimistic on this GPU by
   its own admission; the `nvidia-smi` table is authoritative.
3. **`--cache-reuse` re-explained.** It was blamed for run-to-run variance; source
   inspection of b10588 shows it is inert in this workload (§4).
4. **The hallucination claim deleted.** `temperature 0.0` does not prevent invented facts,
   and the "guarantees the exact same JSON" claim was measured false. Correctness is
   credited to the grammar and `verify.go`.
5. **The recall axis added.** The old justification table covered only `temperature`,
   `top_p` and `repeat_penalty` - none of which affect recall - and never mentioned chunk
   size or output headroom. That omission was its largest defect.

### One criticism that has since been withdrawn

An earlier revision of this file also dismissed `hardware-recommendations.md` for being
"written for Ollama - a runtime this project does not use". **The project now uses
Ollama**, so that objection is void, and its Modelfile-based approach was closer to the
current design than the criticism allowed.

Recorded rather than quietly edited out. The five corrections above were about wrong
numbers and false claims; the runtime complaint was about a choice that has since been
reversed, and the two should not be confused.

---

## 10. Re-validating this file

After a driver update, an Ollama upgrade, or a model change:

```powershell
llmfit system
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
llmfit info "Qwen/Qwen2.5-7B-Instruct-1M"
llmfit plan "Qwen/Qwen2.5-7B-Instruct-1M" --context 16384 --kv-quant q8_0
ollama --version
ollama list
```

**Confirm the service environment first** (§4) — without these the budget in §3 does not
hold:

```powershell
[Environment]::GetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', 'User')
[Environment]::GetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'User')
[Environment]::GetEnvironmentVariable('OLLAMA_NUM_PARALLEL', 'User')
```

To re-measure the footprint, load the model and read the card while it is resident. The
`PROCESSOR` column must read `100% GPU`:

```powershell
ollama run fact-extractor "warm up" --keepalive 5m
ollama ps
nvidia-smi --query-compute-apps=process_name,used_memory --format=csv
ollama stop fact-extractor
```

To replace llmfit's estimated speed with a real number, read `eval_count` and
`eval_duration` from a non-streaming response:

```powershell
$body = '{"model":"fact-extractor","prompt":"Count to fifty.","stream":false,"options":{"num_ctx":16384,"seed":42}}'
$r = Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'
"{0:N1} tok/s" -f ($r.eval_count / ($r.eval_duration / 1e9))
```

Compare against the 25-32 tok/s already recorded.
