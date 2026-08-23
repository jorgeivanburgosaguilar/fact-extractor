// Command fact-extractor turns one text file into a structured, source-traceable
// list of facts and exits.
//
// It does one job. Model, context and sampling come from settings.json; the
// input and output filenames are fixed. The only flag is --benchmark.
//
// The Ollama service is managed for the one-shot flow: a service already
// running is adopted and left alone; otherwise one is started with the tuning
// environment from settings.json and stopped on exit, Ctrl-C included. Every
// request carries keep_alive: 0, so the model is unloaded as soon as the run
// finishes and no VRAM is left held.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"fact-extractor/internal/benchmark"
	"fact-extractor/internal/facts"
	"fact-extractor/internal/ollama"
	"fact-extractor/internal/service"
	"fact-extractor/internal/settings"
	"fact-extractor/internal/textsplit"
	"fact-extractor/internal/validate"
)

const version = "3.0.0"

// Compiled-in copies of the files the tool reads, so a bare executable beside a
// settings.json still works. A file of the same name next to the exe wins.
var (
	//go:embed schemas/facts.json
	defaultSchema []byte
	//go:embed system-instruction.md
	defaultSystemInstruction []byte
)

const (
	instructionFile = "system-instruction.md"
	promptFile      = "prompt.md"
	outputFile      = "result.json"
	corpusDir       = "corpus"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	bench := flag.Bool("benchmark", false, "score the gold corpus instead of processing prompt.md")
	coder := flag.Bool("coder", false, "use ollama.coder_model instead of ollama.model")
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() > 0 {
		usage()
		return fmt.Errorf("unexpected argument %q", flag.Arg(0))
	}

	baseDir, err := resolveBaseDir()
	if err != nil {
		return err
	}
	cfg, err := settings.Load(baseDir)
	if err != nil {
		return err
	}

	system, err := readInstruction(cfg)
	if err != nil {
		return err
	}
	schema, err := readSchema(cfg)
	if err != nil {
		return err
	}

	// The service must not outlive this call when we are the ones who started
	// it — including on Ctrl-C. An adopted service is never stopped: it is not
	// ours to take down.
	svc, err := ensureService(cfg)
	if err != nil {
		return err
	}
	defer svc.Stop()
	stopOnInterrupt(svc)

	model, err := chooseModel(cfg, *coder)
	if err != nil {
		return err
	}
	client := ollama.New(cfg.Ollama.Host, model)
	if err := client.Preflight(); err != nil {
		return err
	}

	ex := &extractor{cfg: cfg, client: client, system: system, schema: schema}

	if *bench {
		return runBenchmark(ex, cfg)
	}
	return runOnce(ex, cfg)
}

// chooseModel picks the model this run will use.
//
// The choice is made once, here, before any request is sent, so a single run
// names exactly one model. That is what keeps the 6 GB budget safe with two
// models installed: there is no code path that can put both in flight.
func chooseModel(cfg *settings.Settings, coder bool) (string, error) {
	if !coder {
		return cfg.Ollama.Model, nil
	}
	if cfg.Ollama.CoderModel == "" {
		return "", fmt.Errorf("--coder needs \"ollama.coder_model\" in settings.json, which is empty.\n" +
			"build.ps1 records it when the coder .gguf is present.")
	}
	return cfg.Ollama.CoderModel, nil
}

// ensureService returns a handle on the Ollama service the run will use.
//
// A service already answering is adopted as-is — with a warning, because its
// environment is unknown and the VRAM budget in hardware-finetune.md may not
// hold there. Otherwise, when settings.json says to manage the service, one is
// started with the tuning environment applied. A nil-cmd handle from either
// path makes the deferred Stop a no-op for anything we did not start.
func ensureService(cfg *settings.Settings) (*service.Handle, error) {
	if service.Detect(cfg.Ollama.Host) {
		if cfg.Service.Manage {
			fmt.Fprintf(os.Stderr,
				"using the Ollama service already running at %s (it will be left running).\n"+
					"  Its environment is unknown here - the VRAM budget assumes\n"+
					"  OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_NUM_PARALLEL=1.\n",
				cfg.Ollama.Host)
		}
		return nil, nil
	}
	if !cfg.Service.Manage {
		return nil, fmt.Errorf("cannot reach Ollama at %s and service.manage is false in settings.json.\n"+
			"Start the service yourself with 'ollama serve', or set service.manage to true.",
			cfg.Ollama.Host)
	}

	fmt.Fprintf(os.Stderr, "starting %s serve ... ", cfg.Service.Command)
	start := time.Now()
	h, err := service.Start(cfg.Ollama.Host, cfg.Service.Command, cfg.Service.Env, cfg.Service.StartupTimeout())
	if err != nil {
		fmt.Fprintln(os.Stderr)
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "up (%s)\n", took(start))
	return h, nil
}

// stopOnInterrupt takes a service we started down if the user aborts a long
// run, rather than stranding runner processes holding ~4.7 GB of VRAM.
func stopOnInterrupt(svc *service.Handle) {
	ch := make(chan os.Signal, 1)
	// Ctrl-C, and also Ctrl-Break or a closing console window, which Windows
	// delivers as SIGTERM.
	signal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ch
		fmt.Fprintln(os.Stderr, "\ninterrupted; shutting down")
		svc.Stop()
		os.Exit(130)
	}()
}

// extractor holds everything that stays the same across chunks and across
// corpus cases.
type extractor struct {
	cfg    *settings.Settings
	client *ollama.Client
	system string
	schema json.RawMessage
}

// stats is what a run reports about itself.
type stats struct {
	chunks   int
	promptN  int
	predictN int
	verified facts.VerifyResult
}

// extract runs the whole pipeline over one text: split, generate per chunk,
// verify every span against the chunk it came from, merge.
func (e *extractor) extract(text string, progress bool) (*facts.Document, stats, error) {
	var st stats

	chunks, err := textsplit.Split(text, e.cfg.ChunkTokens, textsplit.EstimateCounter)
	if err != nil {
		return nil, st, err
	}
	st.chunks = len(chunks)
	if err := e.checkFits(chunks); err != nil {
		return nil, st, err
	}
	if progress && len(chunks) > 1 {
		fmt.Fprintf(os.Stderr, "input split into %d chunks of up to %d tokens\n",
			len(chunks), e.cfg.ChunkTokens)
	}

	// The system message is sent first and byte-identical on every request, so
	// it stays the cached common prefix and chunks 2..N do not re-prefill it.
	sys := ollama.Message{Role: "system", Content: e.system}

	var docs []*facts.Document
	for i, chunk := range chunks {
		if progress && len(chunks) > 1 {
			fmt.Fprintf(os.Stderr, "  chunk %d/%d ... ", i+1, len(chunks))
		}
		res, err := e.client.Chat(
			[]ollama.Message{sys, {Role: "user", Content: chunk}},
			e.cfg.Options, e.schema, e.cfg.Ollama.KeepAlive)
		if err != nil {
			return nil, st, fmt.Errorf("chunk %d/%d: %w", i+1, len(chunks), err)
		}
		st.promptN += res.PromptN
		st.predictN += res.PredictedN

		if progress {
			if len(chunks) > 1 {
				fmt.Fprintf(os.Stderr, "%d tokens (%.1f tok/s)\n", res.PredictedN, res.PerSecond)
			} else {
				fmt.Fprintf(os.Stderr, "generated %d tokens (%.1f tok/s)\n",
					res.PredictedN, res.PerSecond)
			}
		}
		if res.Truncated() {
			return nil, st, e.truncated(res.Content, i, len(chunks))
		}

		doc, err := facts.Parse(res.Content)
		if err != nil {
			return nil, st, e.rawFallback(res.Content, err)
		}
		// Check every citation against the text it came from, so verbatim is
		// always either null or a real substring.
		v := facts.Verify(doc, chunk)
		st.verified.Exact += v.Exact
		st.verified.Repaired += v.Repaired
		st.verified.Dropped += v.Dropped
		docs = append(docs, doc)
	}

	return facts.Merge(docs), st, nil
}

// checkFits refuses to send a request that cannot fit the loaded context, which
// would otherwise fail deep inside the runtime with an opaque error.
//
// Chunk sizes are estimated rather than tokenized: Ollama exposes no public
// tokenizer, so the headroom below is deliberately generous.
func (e *extractor) checkFits(chunks []string) error {
	sysN := textsplit.Estimate(e.system)
	largest := 0
	for _, c := range chunks {
		if n := textsplit.Estimate(c); n > largest {
			largest = n
		}
	}
	// Leave room for the reply; a fact list can be as long as its source.
	needed := sysN + largest + largest/2 + 512
	ctx := e.cfg.NumCtx()
	if needed <= ctx {
		return nil
	}
	return fmt.Errorf(
		"input does not fit: system %d + largest chunk %d tokens needs about %d, num_ctx is %d\n"+
			"Lower \"chunk_tokens\" in settings.json to %d, or raise \"options.num_ctx\".",
		sysN, largest, needed, ctx, e.cfg.ChunkTokens/2)
}

// runOnce is the normal path: prompt.md in, result.json out.
func runOnce(e *extractor, cfg *settings.Settings) error {
	path := cfg.Resolve(promptFile)
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	text := string(raw)
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("%s is empty", path)
	}

	started := time.Now()
	doc, st, err := e.extract(text, true)
	if err != nil {
		return err
	}

	out, err := facts.Encode(doc)
	if err != nil {
		return err
	}
	dest := cfg.Resolve(outputFile)
	if err := os.WriteFile(dest, out, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", dest, err)
	}

	fmt.Fprintf(os.Stderr, "extracted %d facts using %s\n", len(doc.Facts), e.client.Model())
	if st.verified.Any() {
		fmt.Fprintf(os.Stderr,
			"citations: %d exact, %d snapped to the source, %d not found and set to null\n",
			st.verified.Exact, st.verified.Repaired, st.verified.Dropped)
	}
	fmt.Fprintf(os.Stderr, "wrote %s  (%d prompt + %d generated tokens, %s)\n",
		dest, st.promptN, st.predictN, took(started))
	return nil
}

// runBenchmark scores every corpus case and fails the process if any case does.
func runBenchmark(e *extractor, cfg *settings.Settings) error {
	dir := cfg.Resolve(corpusDir)
	cases, err := benchmark.LoadCases(dir)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "benchmarking %s against %d corpus case(s)\n",
		e.client.Model(), len(cases))
	fmt.Fprintf(os.Stderr, "a case passes with at most %d gold facts unmatched\n",
		benchmark.Threshold)
	checkPlacement(e)
	fmt.Fprintln(os.Stderr)

	started := time.Now()
	failed := 0

	for _, c := range cases {
		raw, err := os.ReadFile(c.Source)
		if err != nil {
			return fmt.Errorf("case %s: reading %s: %w", c.Name, c.Source, err)
		}
		text := string(raw)

		fmt.Fprintf(os.Stderr, "%s\n", c.Name)
		doc, st, err := e.extract(text, false)
		if err != nil {
			return fmt.Errorf("case %s: %w", c.Name, err)
		}

		r := benchmark.Score(c.Name, doc, text, c.Gold)
		report(r, st, len(c.Gold))

		// The independent second opinion: the same contract check the standalone
		// checkfacts tool makes, run on the merged document against the whole
		// source. Reported alongside the score, never part of pass/fail.
		if enc, err := facts.Encode(doc); err == nil {
			if v, err := validate.Check(enc, text); err != nil {
				fmt.Fprintf(os.Stderr, "    validation error: %v\n", err)
			} else if !v.OK() {
				fmt.Fprintf(os.Stderr, "    validation: %d contract violation(s):\n", len(v.Problems))
				for _, p := range v.Problems {
					fmt.Fprintf(os.Stderr, "      %s\n", p)
				}
			} else {
				fmt.Fprintf(os.Stderr, "    validation: contract OK (%d/%d spans exact)\n", v.Exact, v.Checked)
			}
		}

		if !r.Pass() {
			failed++
		}
	}

	fmt.Fprintf(os.Stderr, "\n%d/%d cases passed in %s\n",
		len(cases)-failed, len(cases), took(started))
	if failed > 0 {
		return fmt.Errorf("%d benchmark case(s) failed", failed)
	}
	return nil
}

// checkPlacement loads the model briefly and reports how it is split between
// GPU and CPU.
//
// A partially offloaded model runs at roughly half speed and says nothing about
// it, so a benchmark that measures one is comparing against the wrong baseline.
// This warns rather than refusing: a smaller custom model may legitimately not
// fit, and the operator is better placed than this code to judge that. What it
// must never do is stay quiet.
func checkPlacement(e *extractor) {
	if err := e.client.Warmup(e.cfg.Options, 30); err != nil {
		fmt.Fprintf(os.Stderr, "note: could not warm the model to check GPU placement: %v\n", err)
		return
	}
	p, err := e.client.Running()
	if err != nil {
		fmt.Fprintf(os.Stderr, "note: could not read GPU placement: %v\n", err)
		return
	}
	if p.Size == 0 {
		fmt.Fprintln(os.Stderr, "note: no model reported loaded; GPU placement unknown")
		return
	}
	if p.FullyOnGPU() {
		fmt.Fprintf(os.Stderr, "placement: %s\n", p)
		return
	}
	fmt.Fprintf(os.Stderr,
		"\nWARNING: the model is only %s.\n"+
			"  Part of it is running on the CPU, which roughly halves throughput and\n"+
			"  makes these results incomparable with a fully resident run.\n"+
			"  Check OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0 are set for\n"+
			"  the service, or lower options.num_ctx. See hardware-finetune.md section 4.\n\n",
		p)
}

func report(r benchmark.Result, st stats, goldN int) {
	verdict := "PASS"
	if !r.Pass() {
		verdict = "FAIL"
	}
	fmt.Fprintf(os.Stderr, "  %s  %d/%d gold facts matched, %d extracted (%d chunk(s))\n",
		verdict, r.Matched, goldN, r.Extracted, st.chunks)

	for _, g := range r.Missed {
		fmt.Fprintf(os.Stderr, "    missed: %q\n", truncate(g.Verbatim, 70))
	}
	for _, g := range r.Unanchored {
		fmt.Fprintf(os.Stderr,
			"    gold span is not in the source, fix the corpus: %q\n", truncate(g.Verbatim, 70))
	}
	if r.TypeMismatch > 0 {
		fmt.Fprintf(os.Stderr, "    %d matched fact(s) classified differently than the gold list\n",
			r.TypeMismatch)
	}
	// The number worth watching when comparing two models.
	if r.Fabricated > 0 {
		fmt.Fprintf(os.Stderr, "    %d citation(s) failed verification\n", r.Fabricated)
	}
	if r.Inferred > 0 {
		fmt.Fprintf(os.Stderr, "    %d inferred fact(s), not graded\n", r.Inferred)
	}
	if st.verified.Repaired > 0 {
		fmt.Fprintf(os.Stderr, "    %d span(s) snapped to the source\n", st.verified.Repaired)
	}
}

// readInstruction prefers the file on disk and falls back to the copy compiled
// into the binary.
func readInstruction(cfg *settings.Settings) (string, error) {
	path := cfg.Resolve(instructionFile)
	raw, err := os.ReadFile(path)
	if err == nil {
		return string(raw), nil
	}
	if os.IsNotExist(err) {
		return string(defaultSystemInstruction), nil
	}
	return "", fmt.Errorf("reading %s: %w", path, err)
}

func readSchema(cfg *settings.Settings) (json.RawMessage, error) {
	raw := defaultSchema
	path := cfg.Resolve(filepath.Join("schemas", "facts.json"))
	if file, err := os.ReadFile(path); err == nil {
		raw = file
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	if !json.Valid(raw) {
		return nil, fmt.Errorf("%s is not valid JSON", path)
	}
	return json.RawMessage(raw), nil
}

func (e *extractor) truncated(content string, i, n int) error {
	path := e.cfg.Resolve("result.raw.txt")
	_ = os.WriteFile(path, []byte(content), 0o644)
	return fmt.Errorf("chunk %d/%d ran out of room, so the output is incomplete\n"+
		"Raw text saved to %s. Lower \"chunk_tokens\" in settings.json.", i+1, n, path)
}

func (e *extractor) rawFallback(content string, cause error) error {
	path := e.cfg.Resolve("result.raw.txt")
	_ = os.WriteFile(path, []byte(content), 0o644)
	return fmt.Errorf("%w\nRaw output saved to %s", cause, path)
}

// resolveBaseDir finds the directory holding settings.json and the data files:
// the one containing the executable, or the working directory when running
// under `go run`, which builds into a temp directory.
func resolveBaseDir() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return os.Getwd()
	}
	dir := filepath.Dir(exe)
	if _, err := os.Stat(filepath.Join(dir, "settings.json")); err != nil {
		if cwd, err := os.Getwd(); err == nil {
			if _, err := os.Stat(filepath.Join(cwd, "settings.json")); err == nil {
				return cwd, nil
			}
		}
	}
	return dir, nil
}

func usage() {
	fmt.Fprint(os.Stderr, `fact-extractor `+version+` - extract source-traceable facts from a text file

Usage: fact-extractor [--benchmark] [--coder]

  (no flags)     `+promptFile+` -> `+outputFile+`
  --benchmark    score the gold corpus in `+corpusDir+`/ and exit 0 or 1
  --coder        use ollama.coder_model instead of ollama.model

The flags combine: --benchmark --coder scores the same corpus under the second
model, which is what the second model is for. Only one model is ever named per
run, so both are never resident at once.

Model, context and sampling come from settings.json, which is produced by the
build. The Ollama service is started if none is running (see settings.json), and
stopped again on exit; a service found already running is used and left alone.
`)
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func took(start time.Time) string {
	d := time.Since(start)
	if d < time.Second {
		return fmt.Sprintf("%d ms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1f s", d.Seconds())
}
