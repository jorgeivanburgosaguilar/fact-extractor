# Hardware Fine-Tuning: Qwen2.5-7B-Instruct-1M on an RTX 3050 6 GB

This file owns every number in this project that exists because of this specific machine:
VRAM budgets, context limits, runtime settings and sampling values. [`AGENTS.md`](AGENTS.md)
says what the tool is and does; this file says what numbers to run it with on this GPU, and
why each one is what it is.

| | |
|---|---|
| Validated | 2026-08-23 |
| GPU driver | 591.66 |
| Runtime | **Ollama** 0.32.15 (llama.cpp underneath) |
| Baseline measurements | llama.cpp **b10588** (`70adb1b4c`), direct — see §1.5 |
| Model | `Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf` (~4.68 GB on disk) |

Every figure below is reproducible with the command printed beside it. Nothing here is an
estimate unless it is labelled as one. §4 lists the commands to re-run after a driver, build
or model change.

---

## 1. The environment

### 1.1 The machine

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

The desktop itself occupies part of the card — observed between **187 MiB** (nothing
running) and **696 MiB** (normal desktop session). Budget against the higher figure.

### 1.2 The runtime

Ollama 0.32.15 runs llama.cpp underneath. Two of its scheduler behaviours matter for
everything that follows (`server/sched.go`):

- On a CUDA OOM the scheduler reduces context and retries
  (`reduceAutoNumCtxForLoadOOM`) — **but only when the context is "auto"**. An explicitly
  supplied `num_ctx` cannot be silently shrunk.
- **Neither** the OOM path nor the model-eviction path ever lowers the user-specified GPU
  layer count.

The baseline VRAM measurements in §1.5 were taken by driving `llama-server` directly at
build b10588, not through Ollama. Ollama adds its own process overhead and its own memory
estimator (§1.6), so the table is the reference for *how context scales*, not an exact
Ollama footprint. `ollama ps` (§1.6) is the check that tells you where you actually stand.

### 1.3 The model, and why this one

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

llmfit says plainly that this GPU is absent from its bandwidth table. Measured against
`nvidia-smi`, its VRAM projections run consistently **~600 MB high**:

| Context | llmfit estimate (q8_0 KV) | Measured `llama-server` | llmfit error |
|---|---|---|---|
| 8192 | 5.14 GB | 4645 MiB (4.54 GB) | +0.60 GB |
| 16384 | 5.35 GB | 4891 MiB (4.78 GB) | +0.57 GB |
| 32768 | 5.79 GB | 5208 MiB (5.09 GB) | +0.70 GB |

Reproduce with `llmfit plan "Qwen/Qwen2.5-7B-Instruct-1M" --context N --kv-quant q8_0`.

**Direct measurement wins over the heuristic here.** The "Marginal / 89.2%" verdict is
computed from the inflated estimate; §1.5 uses the measured figures instead. llmfit is
authoritative on model *ranking* — it correctly picks this model over alternatives for this
task — and is guessing at this specific GPU's *capacity*. A score belongs to one
model/quantization/runtime triple; a GPTQ or AWQ build scored on vLLM does not transfer to a
Q4_K_M GGUF on llama.cpp, even for the "same" base model.

The model's trained context is **1010000**, far above any window this project uses
(`ollama show <model>` reports it) — the window in §2.2 is never the model's limit. A model
trained below the configured `num_ctx` would degrade silently rather than refuse, which is
worth checking before swapping models.

### 1.4 The binding constraint: 6 GB

6144 MiB of VRAM against a 4.68 GB model file. One 7B model fits at a time, with roughly
1 GB left over for the KV cache and compute buffers. **There is no configuration in which
two models are loaded together.** Every decision in §2 is downstream of this sentence.

### 1.5 Measured VRAM against context

Measured with `nvidia-smi` while `llama-server` held the model, desktop idling at 696 MiB:

| Configuration | `llama-server` | Total incl. desktop | Verdict |
|---|:---:|:---:|:---|
| `-c 4096` | 4523 MiB | 5219 MiB | fits |
| `-c 8192` | 4645 MiB | 5341 MiB | fits |
| **`-c 16384`** | **4891 MiB** | **5587 MiB** | **default — comfortable** |
| `-c 32768` | 5208 MiB | 5904 MiB | practical ceiling, ~0.24 GB spare |
| `-c 65536` | 5176 MiB | 5872 MiB | **spilled to system RAM** |

Two constants come out of this table and are cited throughout §2: KV cache measures
**~30 KB/token** with `q8_0` across the 4k–16k range, and going from 8192 to 16384 costs
**246 MiB**.

### 1.6 Two silent failure modes — and how to detect them

Both of these degrade throughput by roughly half **with no error anywhere in the log.** A
silent halving is harder to notice than a crash, which is why detecting them matters more
than any single tuning value.

**(a) WDDM spill.** On this driver, exceeding VRAM does not abort — the allocation spills
into system RAM. Measured at `-c 65536`:

| | Within budget | Spilled |
|---|---|---|
| Generation | 31.4 tok/s | **16.2 tok/s** |
| Prefill | ~1200 tok/s | **272 tok/s** |

**Rule:** if an allocation failure is ever reported, reduce context first, never `num_gpu`.
Moving layers to the CPU costs more speed than halving the context — and per §1.2 the
scheduler already enforces exactly this for automatic context.

**(b) Ollama's own VRAM estimate assumes f16.** `PredictServerVRAM` in `llm/llama_server.go`
computes the KV cache as:

```go
// KV cache: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (f16)
kvCache := 2 * layers * kvHeads * headDim * uint64(numCtx) * 2
```

The trailing `2 bytes` is **hardcoded f16**, even when `OLLAMA_KV_CACHE_TYPE=q8_0` (§2.3)
halves the real cost to 1 byte per element. Ollama therefore over-estimates this model's
footprint and may decide to leave layers on the CPU that would have fitted on the GPU — the
same class of failure as (a), reached by a different route.

**Detection for both.** `ollama ps` reports the actual split:

```
NAME              ID              SIZE      PROCESSOR    UNTIL
fact-extractor    a1b2c3d4e5f6    5.2 GB    100% GPU     4 seconds from now
```

`100% GPU` is the required state. Anything like `30%/70% CPU/GPU` means layers were
offloaded and any benchmark taken in that state is measuring the wrong thing.

**This project's own placement check only runs under `--benchmark`**
(`main.go`, `checkPlacement`) — a normal extraction run never reports placement. Run
`ollama ps` by hand after any tuning change, and before trusting any throughput number.

### 1.7 Where these values live

There is no compiled-in default for any tunable. The **sole producer of `settings.json` is
`build.ps1`** — `internal/settings` only unmarshals what is on disk and validates it
(`num_ctx > 0`, `chunk_tokens > 0`, non-empty `options`). To change a value, edit
`build.ps1` and re-run it (§4).

| Layer | File | Role |
|---|---|---|
| Generator | `build.ps1` (`options` block, service env block) | Where a human edits a value |
| Artifact | `dist/settings.json` | What the CLI reads |
| Consumer | `internal/settings/settings.go` | Unmarshals and validates |
| Sender | `internal/ollama/client.go` | Puts `options` on the wire unchanged |

`options` is an open map (`map[string]any` in `settings.go`), so any extra key placed there
in `settings.json` is forwarded to Ollama untouched, whether or not this document mentions
it.

**Request options vs. service environment — the split that matters most.** The `options`
block travels on every request; §2.3's four `OLLAMA_*` variables are fixed once, when the
Ollama service starts. With `service.manage: true` (the shipped default) the CLI applies
them itself from `settings.json`'s `service.env` block, so `settings.json` owns the whole
tuning story. When Ollama is already running, it is **adopted, never restarted** — its
environment is unknown, and the CLI can only warn.

### 1.8 Gemma 4 E2B QAT — the reasoning-model comparison

Reachable with `--model fact-extractor-gemma4-qat`, thinking on by default (`README.md` has
the resulting scoreboard) — the historical entry now; §1.9 measures the non-QAT release that
overtook it and backs the default `fact-extractor` name today. Measured **2026-09-09**, same
machine as §1.1, **Ollama 0.33.3** rather than the 0.32.15 this file was otherwise validated
against (§4) — noted because item 7 of §3 predicted this experiment before it was run; these
are the numbers that landed.

```
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv
2161 MiB, 6144 MiB
```
Measured with the model warm (`keep_alive: 30`) at `num_ctx 16384`, desktop included — the
same methodology as §1.5, on a card otherwise idle. Against Qwen's 5587 MiB at the same
context (§1.5), Gemma leaves roughly **3.4 GB more headroom** — expected, since its file is
3.35 GB against Qwen's 4.68 GB, and `ollama ps` reports the whole loaded model at
1,551,850,535 bytes resident on GPU (`size == size_vram`, 100% GPU, confirmed independently
by `--benchmark`'s `checkPlacement` preflight on every run in the scoreboard).

**The GGUF itself:** `general.architecture = gemma4`, `general.size_label = 4.6B`,
`gemma4.context_length = 131072` — far above `num_ctx 16384` (§2.2), so no risk of the silent
degradation §1.3 warns a model trained below the configured context would show. A sibling
`mmproj-gemma-4-E2B-it-QAT-BF16.gguf` ships alongside the text weights; it is the vision
projector and is **not used** — the Modelfile (`FROM` only, §2's rule) never references it.

**Thinking, measured directly, before any benchmark ran.** A probe sent the real
`system-instruction.md`, the real `schemas/facts.json` as `format`, and `corpus/01-news.md`
to the live model three ways — `think` omitted, `true`, `false` — reading `message.thinking`
and `message.content` back separately on each:

| `think` sent | `eval_count` | thinking trace | facts extracted | leaked into `content`? |
|---|---:|---:|---:|---|
| omitted | 3442 | 6352 chars | 26 | no |
| `true` | 3442 (identical — greedy, seeded) | 6352 chars | 26 | no |
| `false` | 1452 | 0 | 19 | no |

Three things this settles: **the model's own default is thinking-on** — omitting `think`
behaves identically to `true`, because decoding is greedy with a fixed seed (§2.4) — so
`main.go` pins `think: true` explicitly by default (only `--no-think` turns it off) rather
than leaving it to that default, the same reasoning §2.2 gives for always sending `num_ctx`.
**Ollama's `gemma4` builtin parser
splits the reasoning channel cleanly**: `message.content` was valid, schema-conforming JSON
in all three variants, never once carrying the `<|think|>` / `<|channel>` markers the raw
chat template (embedded in the GGUF) uses to delimit thinking — confirmed again, per-chunk,
across the full five-case benchmark. **Thinking measurably changes recall, not just token
spend**, on this one document: 26 raw facts with thinking against 19 without, before dedup or
verification.

**The measured cost predicted in §3 item 7.** `eval_count` (generation tokens, thinking and
content combined — there is no separate accounting) went from 1452 to 3442 for the same
chunk: thinking on costs **~2.4×** the generation tokens here. Across the full benchmark,
thinking traces ran 4502–20556 characters per case (`05-survey`'s three-chunk case spent the
most, 20556, consistent with it being the longest document). Generation throughput measured
**~58 tok/s** with thinking on and the model fully GPU-resident (3695 generated tokens at
58.4 tok/s, then 171 at 58.3 tok/s, single-document run) — faster than Qwen's 25–32 tok/s
per-token rate, as expected for a much smaller model, but the token *volume* thinking adds
works against that rate: the five-case benchmark with thinking on took 581.3 s wall time,
against a comparable Qwen benchmark run typically finishing in well under half that.

### 1.9 Gemma 4 E2B (non-QAT) — the version that overtook it

Reachable with `--model fact-extractor-gemma4-qat` for the QAT sibling from here on; this
build backs the default `fact-extractor` name. Measured **2026-09-09**, same machine and
Ollama version as §1.8. `.gguf`: `gemma-4-E2B-it-Q4_K_M.gguf`, 3.43 GB on disk (3,427,880,384
bytes) — QAT (quantization-*aware training*, a checkpoint fine-tuned to be robust under
quantization) and this post-training Q4_K_M quantization of the base release are genuinely
different weight sets, not two quant schemes of the same checkpoint, so this section repeats
§1.8's measurements rather than assuming they carry over.

```
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv
2536 MiB, 6144 MiB
```
Same methodology as §1.5/§1.8: model warm (`keep_alive: 30`), `num_ctx 16384`, desktop
included, card otherwise idle, measured under the tuned service environment (§2.3) rather
than an ad-hoc `ollama serve`. Against QAT's 2161 MiB, this build costs **~375 MiB more** —
in the direction its 80 MB-larger file predicts. `ollama ps` / `/api/ps` report the whole
loaded model at 1,630,210,619 bytes resident on GPU (`size == size_vram`, 100% GPU, confirmed
by `--benchmark`'s `checkPlacement` preflight on every run in the scoreboard) — against QAT's
1,551,850,535, a ~78 MB gap that tracks the file-size difference almost exactly.

**The GGUF itself, and thinking, carry over unchanged from §1.8:** `general.architecture =
gemma4`, `general.size_label = 4.6B`, `gemma4.context_length = 131072`. `/api/show` reports
the same three capabilities as QAT — `completion`, `tools`, `thinking` — so
`ollama.Client.SupportsThinking` (AGENTS.md §2) turns thinking on by default here with no
special-casing; nothing in `main.go` or `internal/ollama` distinguishes this model from its
QAT sibling. The sibling `mmproj-gemma-4-E2B-it-BF16.gguf` is the vision projector and is,
as with QAT, not referenced by the Modelfile.

**llmfit, checked before building anything.** Its database has no entry for the QAT release
— only community re-quants of it turn up — but does carry the canonical non-QAT release:

```
$ llmfit info "google/gemma-4-E2B-it"
Overall Score: 86.1/100   Quality 76  Speed 100  Fit 85  Context 100
Fit Analysis: Good, Memory Utilization 81.2% (4.9/6.0 GB)

$ llmfit plan "google/gemma-4-E2B-it" --context 16384 --kv-quant q8_0
Minimum Hardware: VRAM 3.7 GB, RAM 8.0 GB, cores 4
Feasible Run Paths: GPU: yes, est. speed 49.4 tok/s
```

The same pattern §1.3 found for Qwen on this GPU repeats here: llmfit's estimate (3.7–4.9 GB)
sits well above the measured 2536 MiB — this GPU is absent from its bandwidth table, and the
"Good" / "81.2%" verdict is computed from that inflated figure. Measured wins over the
heuristic, same as everywhere else in this file (§1.3); llmfit's job here was ranking-level
reassurance that the model fits *at all* before spending the time to `ollama create` and
benchmark it, not the authoritative VRAM number.

**The benchmark win, not just a tie.** 85/97 (88%) against QAT's 83/97 (86%) — a clean win on
`05-survey` (18/23 against QAT's 16/23) and an exact tie on the other four documents. Per
AGENTS.md §2's rule (the current benchmark winner backs the default name), this model now
does; QAT moved to `fact-extractor-gemma4-qat`, a historical entry kept for re-scoring after
`system-instruction.md` or the corpus changes. Full per-document numbers are in
[`README.md`](README.md#the-scoreboard).

**Faster, too — not a speed-for-accuracy trade.** Generation measured **~69 tok/s** (2986
tokens at 68.8 tok/s, then 210 at 69.1 tok/s, single-document run, tuned service environment)
against QAT's ~58 tok/s, both fully GPU-resident on the same corpus document and options. The
five-case benchmark finished in 391.8 s wall time, against 470.5 s for QAT in the same
session (§1.8's own 581.3 s figure was measured on an earlier Ollama version, so the 470.5 s
comparison here is the fairer one). Thinking-trace length per case (3151–15921 characters)
also ran shorter than QAT's 4502–20556 on the same corpus — this build reaches its answers
with less reasoning spent, not just faster tokens.

---

## 2. The parameters we set, and why

### 2.1 At a glance

| Parameter | Value | Set in | Why (short form) |
|---|:---:|---|---|
| `num_ctx` | `16384` | `build.ps1` → `options` | Output headroom for the fact list, not input room; ~246 MiB over 8192, no speed cost |
| `num_gpu` | `99` | `build.ps1` → `options` | "more layers than the model has" — full GPU residency |
| `num_batch` | `512` | `build.ps1` → `options` | llama.cpp default, carried over untuned |
| `temperature` | `0.0` | `build.ps1` → `options` | Greedy decoding — buys stability, not correctness |
| `top_k` | `1` | `build.ps1` → `options` | Redundant with `temperature 0`, explicit anyway |
| `top_p` | `1.0` | `build.ps1` → `options` | No-op under greedy decoding — documented so it isn't mistaken for a safeguard |
| `repeat_penalty` | `1.0` | `build.ps1` → `options` | Disabled: a fact list legitimately repeats tokens |
| `seed` | `42` | `build.ps1` → `options` | Pins the sampler; does not make runs bit-identical |
| `chunk_tokens` | `1000` | `build.ps1` → top level | The only real speed/recall tradeoff |
| `keep_alive` | `0` | `build.ps1` → `ollama` | Unload after every run; 6 GB has no room for an idle resident model |
| `format` | `schemas/facts.json` | sent per request | GBNF-constrained JSON — the actual correctness guarantee for *shape* |
| `num_predict`/`max_tokens` | **not sent** | — | An explicit cap could only make the model stop early — the exact failure being avoided |
| `OLLAMA_FLASH_ATTENTION` | `1` | `build.ps1` → service env | Required for the q8_0 KV path |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | `build.ps1` → service env | The measured §1.5 budget depends on this; fp16 doubles KV cache |
| `OLLAMA_NUM_PARALLEL` | `1` | `build.ps1` → service env | One slot — nothing to share on 6 GB, single-request CLI |
| `OLLAMA_MODELS` | `D:\Modelos\Ollama` | `build.ps1` → service env | Local blob-store path, not a tuning decision — change this per machine |

The rest of this section is the long-form justification behind each row.

### 2.2 Context and placement

**`num_ctx: 16384`**

- It costs **246 MiB** over 8192 (§1.5) and **no measurable speed**: prefill and generation
  scale with tokens actually used, not with the window reserved. Unused context is idle
  VRAM, not wasted compute.
- What the room is *for* is **output headroom**, not longer input. At `num_ctx 8192` with
  1000-token chunks the budget is roughly 1,700 (system instruction) + 1,000 (chunk) +
  reply + 512, which boxes in the fact list. At 16384, with chunks held at 1000 or lowered,
  the entire gain goes to letting the model emit as many facts as the passage actually
  contains.
  **16384 buys the model room to *write* a long fact list. It must not buy it a longer
  passage to *read*.** The two rising together is what silently costs recall (§2.6).
- It is sent **explicitly on every request**, because per §1.2 Ollama can only silently
  shrink an *automatic* context. Silent degradation is the failure this whole document
  exists to prevent.

The ceiling: 32768 fits with ~0.24 GB spare (§1.5) but leaves nothing for the desktop's
variable share; 65536 spills to system RAM (§1.6a).

**`num_gpu: 99`** — "all layers on the GPU." Partial offload is roughly half speed (§1.6).
99 is the conventional "more layers than the model has" value, so every layer that can be
placed on the GPU is. Never lower this to solve a memory problem; lower `num_ctx` instead
(§1.6a).

**`num_batch: 512`** — llama.cpp's own default, carried over deliberately rather than
tuned. Raising it to 1024 would trade a little VRAM for prefill throughput; that is
**untested on this machine** and not adopted without measurement first.

### 2.3 The service environment

| `llama-server` flag | Ollama equivalent | Set where |
|---|---|---|
| `-c 16384` | `num_ctx` | `settings.json` → `options` |
| `-ngl 99` | `num_gpu` | `settings.json` → `options` |
| `-b 512` `-ub 512` | `num_batch` | `settings.json` → `options` |
| `-fa on` | `OLLAMA_FLASH_ATTENTION=1` | **service environment** |
| `-ctk q8_0` `-ctv q8_0` | `OLLAMA_KV_CACHE_TYPE=q8_0` | **service environment** |
| `--parallel 1` | `OLLAMA_NUM_PARALLEL=1` | **service environment** |

- **`OLLAMA_KV_CACHE_TYPE=q8_0`** — the load-bearing one. The measured 4891 MiB at 16384
  (§1.5) **depends on it**: at fp16 the KV cache doubles, ~30 KB/token becomes ~56, and the
  entire budget in §1.5 no longer holds.
- **`OLLAMA_FLASH_ATTENTION=1`** — required for the q8_0 KV path to be used efficiently;
  part of the same measured configuration as above.
- **`OLLAMA_NUM_PARALLEL=1`** — one request slot. Parallel slots each reserve their own
  share of the context window; with 6 GB there is nothing to share, and this is a
  single-user CLI that sends one request at a time regardless.
- **`OLLAMA_MODELS=D:\Modelos\Ollama`** — points the model blob store at a drive with room
  for the ~4.7 GB the model occupies a second time after `ollama create`. Read from the
  registry at build time and only emitted if set. This is the one variable in this list
  that is a **local path, not a tuning decision** — change it per machine, it does not
  travel with the rest of this document.

**When Ollama is run independently** — `service.manage: false`, or a service that was
already up and got adopted — these variables must be set for that service by hand; the CLI
can only warn. On Windows, set and restart it:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_NUM_PARALLEL 1
```

### 2.4 Sampling

Sent per request in Ollama's `options` block. Never baked into the Modelfile — see §2.7.

| Parameter | Value | What it actually buys |
|---|:---:|---|
| `temperature` | `0.0` | Greedy decoding — the highest-probability token every step. Buys **stability**, not correctness (§2.8). |
| `top_k` | `1` | Greedy. Redundant with `temperature 0`, set explicitly so the intent isn't left to be inferred. |
| `top_p` | `1.0` | **A no-op.** Under greedy decoding the argmax is taken regardless of the nucleus. Documented only so nobody mistakes it for a safeguard. |
| `repeat_penalty` | `1.0` | Disabled, correctly. Not because it would corrupt JSON structure — the grammar makes that structurally impossible — but because a fact list legitimately repeats tokens (field names, recurring entities), and penalising them skews extraction. |
| `seed` | `42` | Pins the sampler. Does **not** make runs bit-identical — see below. |

**Run-to-run variance is real and expected.** Two runs over the same input produced 17 and
18 facts. This is normal for batched GPU inference — floating-point reduction order is not
fixed, and the amount of cached prefix changes batch composition — and is not worth
engineering against:

- **17 instead of 18** — one marginal claim landed on the other side of a near-tie. Both
  sets fully verified. **Not a defect.**
- **10 instead of 18** — the model *stopped extracting*. Schema-valid, fully verified, and
  missing a third of the document. **A real, silent defect** — what §2.6 exists to prevent.

Never write a test that asserts an exact fact count. Assert traceability and a plausible
floor instead.

### 2.5 Output shape and headroom

- **`format` = `schemas/facts.json`** — not an `options` entry; Ollama's top-level `format`
  field, sent as a full JSON Schema. Ollama forwards it to llama.cpp as `json_schema`, so
  the output is GBNF-constrained: this guarantees parseable JSON and closes `type` to the
  seven allowed categories. It guarantees a **valid** document, never a **complete** one —
  `]` is a legal token after any finished fact object. The grammar enforces shape; it cannot
  enforce exhaustiveness.
- **`num_predict` is deliberately absent** — no `num_predict`/`max_tokens` is sent anywhere
  in this project. Output length is bounded only by what remains of `num_ctx` (§2.2), which
  is the point: an explicit cap could only make the model stop *earlier*, and early stopping
  is exactly the failure being avoided. Truncation is still caught — `done_reason ==
  "length"` makes the CLI write `result.raw.txt` and abort rather than silently emit a
  partial fact list. A model that simply *finishes early* is not truncation, and nothing
  flags that; §2.6 is about reducing how often it happens.
- **`keep_alive: 0`** — unload the model as soon as the run ends, holding no VRAM. On 6 GB,
  a resident model blocks everything else on the machine between runs. The one exception is
  the internal warmup call, which uses a short `keep_alive` so the placement check in §1.6
  has a live model to read.
- **The system message is byte-identical on every request**, so it stays the cached prompt
  prefix. This is the caching behaviour that actually matters here — see §2.7 on
  `--cache-reuse` for the one that doesn't.

### 2.6 Chunking — `chunk_tokens: 1000`

The recall dial, and the only knob that genuinely trades speed against mistakes. Causes of
under-extraction, ranked by how much they bite on this setup:

1. **Chunk size** — dominant. Recall falls as the passage lengthens; attention dilutes and
   mid-chunk facts get skipped.
2. **Output headroom** — a fact list can be as long as its source (§2.5).
3. **The grammar permits stopping** — valid is not the same as complete (§2.5).
4. **The system instruction** — "extract EVERY fact" carries real weight, but is the
   weakest of the four.

What is *absent* from this list matters too: `temperature`, `top_p` and `repeat_penalty`
have no effect on recall at all.

Default is **1000**. On the corpus suite this ties or beats 3000 for recall while giving a
dense multi-chunk document (a citation-heavy survey, say) its evenest possible packing —
raising it to 1200 produces the same request count with a worse split, and only past 1500
does the request count actually drop. 3000 was the previous default; see below.

| Direction | Effect |
|---|---|
| **Lower** (600–800) | Higher recall, more requests. Below roughly 650 a short corpus document (`02-research`, 618 est. tokens) starts to split and gains a runt chunk; going much lower risks splitting `04-code-claims` (480 est. tokens), whose gold pairs must stay in one chunk (section 5 of AGENTS.md). |
| **Default** (1000) | Balanced; the evenest packing available for a dense multi-chunk document. |
| **Higher** (1500–3000) | Fewer requests, at rising risk of silent omission from dense passages. 3000 was the previous default, still reasonable for short documents that fit in one chunk regardless. |

Chunk boundaries follow paragraphs, then sentences, then a hard split. Per-chunk fact lists
are then merged: duplicates are dropped (`facts.Merge`, keyed on the normalised fact
sentence — lowercase, letters and digits only, so punctuation or spacing differences don't
create a second entry) and ids are renumbered 1..N over what remains.

**Chunk size bounds recall, not elapsed time.** Nothing here sends `num_predict` (§2.5), so
the generation ceiling on any one request is whatever `num_ctx` leaves after the prompt:
roughly 13,300 tokens of write room per chunk at 1000, versus roughly 11,350 at 3000. A run
that is slow because generation keeps going to that ceiling — a plausible cause on a
citation-dense, list-shaped document at greedy decoding with `repeat_penalty 1.0` — is not
fixed by a smaller chunk; a smaller chunk mostly wins by being a more tractable per-request
task that is more likely to stop on its own. The lever that actually bounds elapsed time is
a `num_predict` entry in `settings.json`'s `options` block — `internal/settings.Options`
already forwards unknown keys verbatim, so adding it needs no code change — but it is not
set here; that is a deliberate choice left to `build.ps1`, not this file.

**Chunk sizing is an estimate, not a measurement.** Ollama does not expose tokenization on
its public API — `/tokenize` exists only on the llama-server subprocess it manages
internally — so `textsplit.Estimate`, a character-based approximation
(`runes*10/36 + 1`, ~3.6 chars/token, deliberately pessimistic), is the only option. This is
why chunks are sized conservatively against the 16384 window: the slack absorbs the
estimate's error. Real token counts come back in each response as `prompt_eval_count`,
enough for reporting but too late for sizing.

Before any request is sent, the CLI checks the fit and refuses rather than risk a silent
overflow: `system tokens + largest chunk + largest chunk / 2 + 512 <= num_ctx`, and suggests
halving `chunk_tokens` if it fails. This is the only context-related guard in the CLI —
there is no separate warning tied to any fixed context threshold, and no `--ctx` flag; the
only flag is `--benchmark`. Context comes solely from `settings.json`.

### 2.7 Settings we deliberately do not use

- **KV cache `q4_0`** — would free roughly another 50% (llmfit: 5.14 GB at 16k), enough to
  reach 32768. **Rejected:** it degrades precisely the citation fidelity this tool exists
  to provide. Not a trade worth making.
- **`num_batch` 1024** — a plausible prefill win, untested here. Not adopted without a
  measurement.
- **`--cache-reuse`** — not exposed by Ollama, and inert in this workload regardless.
  `--cache-reuse N` (llama.cpp b10588, `tools/server/server-context.cpp`) only attempts KV
  shifting *past* the point where the prompt prefix diverges from what is cached
  (`get_common_prefix()`, loop starting at `head_c = n_past`). Every request here is
  `[system instruction][chunk N]`; the system instruction *is* the common prefix, and it is
  already reused by ordinary prompt caching with no shifting involved. For `--cache-reuse`
  to fire at all, two different chunks would have to share hundreds of consecutive identical
  tokens, which prose does not do. Nothing is lost by Ollama not exposing it.
- **`PARAMETER` lines in the Modelfile** — `ollama create` accepts them, and they become
  defaults that request options then override: two sources of truth for one value, which is
  exactly the drift this file exists to prevent. The Modelfile holds only `FROM
  "<gguf path>"`; `settings.json` owns every parameter.
- **Raising `chunk_tokens` alongside `num_ctx`** — the failure described in §2.2. The extra
  context window is output headroom, not licence to feed the model longer passages.

### 2.8 What none of these parameters buy

Three separate mechanisms do three separate jobs:

| Property | Mechanism |
|---|---|
| Output is valid JSON, categories closed to the 7 allowed | **GBNF grammar** (`format` / `json_schema`, §2.5) |
| Output is stable run to run | `temperature 0` + `top_k 1` — approximately |
| Claims are traceable to the source | **`internal/facts/verify.go`** |

**No sampling parameter on this list prevents hallucination.** `temperature 0.0` makes the
model pick its *most likely* token, with total confidence. If the most likely continuation
is wrong, greedy picks it anyway — and picks it every time. Recorded example: the model
emitted "**Eleven** of the 74 stations" where the source reads "**Twelve**," at temperature
0.0.

What catches that is the verification pass. The CLI holds the source text, so it checks
every `verbatim` span against it: a traceable span is snapped to the exact source
characters, and a span that is not present becomes `null`. After that pass, `verbatim` is by
construction either `null` or a real substring of the input.

The verifier does not touch `confidence` — a failed citation stays distinguishable from an
honest inference, which is what makes the output readable:

| `verbatim` | `confidence` | Meaning |
|---|---|---|
| a span | any | Verified — a real substring of the source |
| `null` | `low` | Inferred from the text, never claimed to be a quote |
| `null` | `high` / `medium` | **A citation that failed verification** |

That third row is the number worth watching when comparing two models: how often a model
fabricates a span outright. Measured on the long-report demo: **62 exact, 2 snapped, 3
dropped** — after which the independent `checkfacts` validator graded all 64 remaining spans
exact.

**This is the tool's actual safety guarantee.** It holds on every run regardless of what the
sampler did, and it is a stronger promise than determinism would have been.

### 2.9 Adapting to other hardware

The rest of this section is tuned for the 6 GB card in §1.1. This table is the per-VRAM
guidance for running the project elsewhere; it is the complement of §3, which asks what we
would do differently *on this machine*.

| Your VRAM | What to change |
|---|---|
| **6 GB** | Nothing. This is the tested configuration. |
| **8–12 GB** | `num_ctx` can go to 32768. Still one 7B model at a time. |
| **16 GB+** | A larger model becomes viable — update `$ggufCandidates` in `build.ps1` and re-check `ollama show`. |
| **4 GB** | Drop `num_ctx` to 8192 (saves ~246 MiB, §1.5) and expect a tighter fit; consider a smaller quant. |
| **No NVIDIA GPU** | Set `num_gpu` to 0 to run on CPU. Correct, but slow. |

Everything tunable lives in `dist\settings.json`, regenerated by every build — so make
lasting changes in `build.ps1`, which writes it (§1.7).

The three service environment variables — `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`,
`OLLAMA_NUM_PARALLEL` — do not change with VRAM tier; see §2.3 for what they do and how to
set them by hand if you run Ollama independently of this CLI.

---

## 3. What better hardware or a reasoning model would change

Each entry: the constraint today, what would change, what it buys, and why it isn't
possible now. Read against the parameter it counters in §2.

1. **6 GB caps the context window** (§1.4, §2.2) → raise `num_ctx` toward 32768–65536 →
   more output headroom per request, and chunk size stops being the tightest recall
   constraint → 32768 leaves only ~0.24 GB spare before the desktop's variable share
   (§1.5), and 65536 spills to system RAM (§1.6a).
2. **6 GB caps the model** (§1.4) → a 14B-class model at Q4_K_M, or the current 7B at
   Q6/Q8 → better extraction judgement and fewer fabricated spans — the `null` +
   `high`/`medium` row in §2.8 → a 14B Q4_K_M is roughly twice the 4.68 GB the current
   weights alone occupy.
3. **KV cache must stay quantised to q8_0** (§2.3, §2.7) → an fp16 KV cache → removes a
   quantisation error term from long-context attention → fp16 doubles KV cache to
   ~56 KB/token and breaks the §1.5 budget outright.
4. **Only one model loaded at a time** (§1.4) → keep an extractor and a
   verifier/critic model resident together → a second pass could re-read the source and
   challenge each fact before output, instead of `verify.go` only checking that a span
   exists → there is no configuration on this GPU in which two models load simultaneously.
5. **~38 GB/s memory bandwidth caps generation at 25–32 tok/s** (§1.1, §4) → a desktop GPU
   with several times the bandwidth → lowering `chunk_tokens` becomes cheap, so the
   higher-recall end of §2.6 becomes affordable without a wall-clock penalty → generation
   here is bandwidth-bound, not compute-bound, so a faster core would not help on its own.
6. **Chunk sizing is an estimate, not a measurement** (§2.6) → a runtime that exposes
   tokenization on its public API (llama-server's own `/tokenize`, or vLLM) → exact chunk
   sizing, removing the conservative slack §2.6 currently spends on estimate error → Ollama
   exposes tokenization only on the llama-server subprocess it manages internally, not
   through its own API.
7. **No reasoning budget** (§2.8) → a model that can think before answering → **tried, not
   just predicted — §1.8 has the measurement.** Gemma 4 E2B, thinking on by default, closes
   `02-research` (14/18 → 18/18) and narrows both remaining capability probes without
   closing them; the mechanism looks like the coverage sweep this row predicted, most
   visible on `02-research`'s qualifiers and appositives coming out as separate facts →
   the predicted cost landed too: thinking tokens compete for the same output headroom §2.2
   fights for (confirmed — no separate accounting, ~2.4× the generation tokens for one
   chunk), though on a smaller, faster model (~58 tok/s vs. 25–32) the wall-clock cost is
   less severe than this row originally worried, even after that overhead. §1.9 pushes this
   further on the same architecture, no extra tuning: the non-QAT release of the same model
   closes `05-survey` further still (13/23 → 16/23 → 18/23) while running faster, not slower
   (~69 tok/s), than the QAT build this row measured.
8. **Greedy decoding is the only stability mechanism available** (§2.4) → with headroom to
   spare, self-consistency — sample several times, keep the facts that appear in a
   majority — would convert the 17-vs-18 variance in §2.4 from noise into an actual
   confidence signal → costs *n* times the generation time, which this card cannot absorb
   today.

§2.9 gives per-VRAM-tier guidance for *running this project elsewhere* (8–12 GB → 32768
context, 16 GB+ → a larger model, no GPU → `num_gpu 0`). This section is the complementary
question — what we would *do differently* here, not how to reach parity somewhere else.

---

## 4. Re-validating this file

After a driver update, an Ollama upgrade, or a model change:

```powershell
llmfit system
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
llmfit info "Qwen/Qwen2.5-7B-Instruct-1M"
llmfit plan "Qwen/Qwen2.5-7B-Instruct-1M" --context 16384 --kv-quant q8_0
ollama --version
ollama list
```

**Confirm the service environment first** (§2.3) — without these the budget in §1.5 does
not hold:

```powershell
[Environment]::GetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', 'User')
[Environment]::GetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'User')
[Environment]::GetEnvironmentVariable('OLLAMA_NUM_PARALLEL', 'User')
[Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
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

Compare against the recorded ranges: generation **25–32 tok/s**, prefill **1100–1400
tok/s**, model load **3.3–5.8 s** from NVMe (about 50 s from USB 3.0).

A value changed here takes effect only after it is edited in `build.ps1` and the script is
re-run (§1.7) — `settings.json` is generated, never hand-edited.
