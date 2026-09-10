package benchmark

import (
	"encoding/json"
	"time"
)

// VerifyCounts mirrors facts.VerifyResult for the JSON record, so this package
// does not need callers to import internal/facts just to read three ints.
type VerifyCounts struct {
	Exact    int `json:"exact"`
	Repaired int `json:"repaired"`
	Dropped  int `json:"dropped"`
}

// ValidationRecord is the internal/validate second opinion for one case,
// reported alongside the score but never part of Pass/Threshold — the same
// rule runBenchmark already follows when it prints this to stderr.
type ValidationRecord struct {
	OK       bool     `json:"ok"`
	Checked  int      `json:"checked"`
	Exact    int      `json:"exact"`
	Problems []string `json:"problems,omitempty"`
	Error    string   `json:"error,omitempty"`
}

// CaseRecord is everything report() prints about one corpus case, in a shape
// that survives a run instead of scrolling past in a terminal.
type CaseRecord struct {
	Name         string            `json:"name"`
	Pass         bool              `json:"pass"`
	Gold         int               `json:"gold"`
	Matched      int               `json:"matched"`
	Missed       int               `json:"missed"`
	Unanchored   int               `json:"unanchored"`
	Extracted    int               `json:"extracted"`
	TypeMismatch int               `json:"type_mismatch"`
	Inferred     int               `json:"inferred"`
	Fabricated   int               `json:"fabricated"`
	Verify       VerifyCounts      `json:"verify"`
	Chunks       int               `json:"chunks"`
	PromptTokens int               `json:"prompt_tokens"`
	EvalTokens   int               `json:"eval_tokens"`
	ThinkChars   int               `json:"think_chars"`
	ElapsedMS    int64             `json:"elapsed_ms"`
	Validation   *ValidationRecord `json:"validation,omitempty"`

	// Glean passes (settings.json "passes" > 1), reported separately so what
	// the extra pass cost can be weighed against what it found. All zero when
	// passes is 1.
	ExtraRequests   int `json:"extra_requests,omitempty"`
	ExtraPromptN    int `json:"extra_prompt_tokens,omitempty"`
	ExtraPredictN   int `json:"extra_eval_tokens,omitempty"`
	ExtraThinkChars int `json:"extra_think_chars,omitempty"`
}

// Totals sums the per-case numbers, so a reader does not have to add them up.
type Totals struct {
	Cases      int `json:"cases"`
	Passed     int `json:"passed"`
	Gold       int `json:"gold"`
	Matched    int `json:"matched"`
	Missed     int `json:"missed"`
	Extracted  int `json:"extracted"`
	Fabricated int `json:"fabricated"`
}

// Run is one whole `--benchmark` invocation: the configuration it ran under,
// every case it scored, and the totals across them. This is reporting only —
// nothing here feeds back into Score, Pass or Threshold.
type Run struct {
	Timestamp   time.Time    `json:"timestamp"`
	Model       string       `json:"model"`
	Think       any          `json:"think"` // nil: this model has no thinking mode; else a bool or effort-level string
	Passes      int          `json:"passes"`
	ChunkTokens int          `json:"chunk_tokens"`
	NumCtx      int          `json:"num_ctx"`
	ElapsedMS   int64        `json:"elapsed_ms"`
	Pass        bool         `json:"pass"`
	Cases       []CaseRecord `json:"cases"`
	Totals      Totals       `json:"totals"`
}

// NewRun folds cases into a Run, computing Totals and the overall Pass so a
// caller cannot forget to keep them in sync with the per-case records.
func NewRun(model string, think any, passes, chunkTokens, numCtx int, elapsed time.Duration, cases []CaseRecord) Run {
	r := Run{
		Timestamp:   time.Now(),
		Model:       model,
		Think:       think,
		Passes:      passes,
		ChunkTokens: chunkTokens,
		NumCtx:      numCtx,
		ElapsedMS:   elapsed.Milliseconds(),
		Cases:       cases,
		Pass:        true,
	}
	r.Totals.Cases = len(cases)
	for _, c := range cases {
		if c.Pass {
			r.Totals.Passed++
		} else {
			r.Pass = false
		}
		r.Totals.Gold += c.Gold
		r.Totals.Matched += c.Matched
		r.Totals.Missed += c.Missed
		r.Totals.Extracted += c.Extracted
		r.Totals.Fabricated += c.Fabricated
	}
	return r
}

// Encode renders a Run as indented JSON.
func Encode(r Run) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
