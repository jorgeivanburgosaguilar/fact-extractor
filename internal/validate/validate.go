// Package validate re-checks a fact document against the contract in
// system-instruction.md: the closed type and confidence vocabularies,
// sequential ids, non-empty atomic facts, duplicates, and — the check the
// grammar cannot make — that every verbatim span really is a substring of the
// source text rather than a paraphrase.
//
// It is deliberately independent of internal/facts. Its value is being a
// second opinion on verify.go: if it shared that code it would confirm the
// same bug rather than catch it. It keeps its own struct definitions and its
// own strict decode.
package validate

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
)

var validTypes = map[string]bool{
	"quote": true, "numeric": true, "event": true, "entity": true,
	"definition": true, "causal": true, "other": true,
}

var validConfidence = map[string]bool{"high": true, "medium": true, "low": true}

type fact struct {
	ID         int     `json:"id"`
	Fact       string  `json:"fact"`
	Type       string  `json:"type"`
	Confidence string  `json:"confidence"`
	Verbatim   *string `json:"verbatim"`
}

type document struct {
	Facts []fact `json:"facts"`
}

// Result is everything one validation pass found.
type Result struct {
	Facts      int
	TypeCounts map[string]int

	// Verbatim accounting.
	Checked         int // spans compared against the source
	Exact           int // copied character-for-character
	Recased         int // real span, capitalization differs
	EdgeDirty       int // real span, stray edge punctuation
	NullInferred    int // verbatim null with confidence low: an honest inference
	FailedCitations int // verbatim null with confidence above low: a citation that failed verification

	Problems []string // contract violations; any of these is a failure
	Warnings []string // traceable but imperfect; reported, never fatal
}

// OK reports whether the document honours the contract.
func (r Result) OK() bool { return len(r.Problems) == 0 }

// Check validates raw JSON against the contract. sourceText may be empty, in
// which case the substring checks are skipped and only the structural rules
// run.
func Check(raw []byte, sourceText string) (Result, error) {
	res := Result{TypeCounts: map[string]int{}}

	var doc document
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return res, fmt.Errorf("not a valid fact document: %w", err)
	}
	res.Facts = len(doc.Facts)

	var haystack string
	if sourceText != "" {
		haystack = squash(sourceText)
	}

	problem := func(f fact, format string, args ...any) {
		res.Problems = append(res.Problems,
			fmt.Sprintf("fact %d: %s", f.ID, fmt.Sprintf(format, args...)))
	}
	warn := func(f fact, format string, args ...any) {
		res.Warnings = append(res.Warnings,
			fmt.Sprintf("fact %d: %s", f.ID, fmt.Sprintf(format, args...)))
	}

	seen := map[string]int{}
	for i, f := range doc.Facts {
		if f.ID != i+1 {
			problem(f, "id is %d but should be %d (1-based, in order)", f.ID, i+1)
		}
		if strings.TrimSpace(f.Fact) == "" {
			problem(f, "fact text is empty")
		}
		if !validTypes[f.Type] {
			problem(f, "type %q is outside the allowed set", f.Type)
		}
		if !validConfidence[f.Confidence] {
			problem(f, "confidence %q is outside the allowed set", f.Confidence)
		}
		res.TypeCounts[f.Type]++

		key := normalize(f.Fact)
		if prev, dup := seen[key]; dup {
			problem(f, "duplicates fact %d", prev)
		} else {
			seen[key] = f.ID
		}

		switch {
		case f.Verbatim == nil:
			// Confidence tells the two null states apart. An inference is the
			// model doing what rule 8 asks; a null span at higher confidence is
			// a citation the verifier had to drop — a model-quality signal worth
			// counting, not a contract violation.
			if f.Confidence == "low" {
				res.NullInferred++
			} else {
				res.FailedCitations++
				warn(f, "citation failed verification: verbatim is null at confidence %q", f.Confidence)
			}
		case haystack != "":
			res.Checked++
			span := squash(*f.Verbatim)
			switch {
			case strings.Contains(haystack, span):
				res.Exact++
			case strings.Contains(strings.ToLower(haystack), strings.ToLower(span)):
				res.Recased++
				warn(f, "verbatim differs from the source only in capitalization: %q", truncate(*f.Verbatim, 70))
			case strings.Contains(strings.ToLower(haystack), strings.ToLower(trimEdges(span))):
				res.EdgeDirty++
				warn(f, "verbatim is a real span with added edge punctuation: %q", truncate(*f.Verbatim, 70))
			default:
				// Nothing resembling this text exists in the source: the model
				// paraphrased instead of quoting. This is the failure that
				// matters, because it breaks traceability.
				problem(f, "verbatim is not a span of the source: %q", truncate(*f.Verbatim, 70))
			}
		}
	}
	return res, nil
}

// Summary renders the one-line accounting used by both checkfacts and the
// benchmark report.
func (r Result) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d facts   ", r.Facts)
	for _, t := range []string{"quote", "numeric", "event", "entity", "definition", "causal", "other"} {
		if r.TypeCounts[t] > 0 {
			fmt.Fprintf(&b, "%s=%d ", t, r.TypeCounts[t])
		}
	}
	fmt.Fprintf(&b, "\nverbatim: %d checked (%d exact), %d inferred, %d failed citation(s), %d re-cased, %d edge punctuation",
		r.Checked, r.Exact, r.NullInferred, r.FailedCitations, r.Recased, r.EdgeDirty)
	return b.String()
}

// trimEdges strips the punctuation and quotation marks a model tends to add
// around a span it is otherwise copying correctly.
func trimEdges(s string) string {
	return strings.Trim(s, " \t\"'“”‘’.,;:!?()[]")
}

// squash collapses whitespace runs so line wrapping cannot defeat substring
// matching.
func squash(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// normalize reduces a fact sentence to a comparison key for duplicate
// detection.
func normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
