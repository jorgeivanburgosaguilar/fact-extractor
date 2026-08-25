// Package validate re-checks a fact document against the contract in
// system-instruction.md: the closed type and confidence vocabularies,
// sequential ids, non-empty atomic facts, duplicates, and — the checks the
// grammar cannot make — that every verbatim span really is a substring of the
// source text rather than a paraphrase, and that the position reported for it
// points at where it actually sits.
//
// It is deliberately independent of internal/facts and internal/output. Its
// value is being a second opinion on verify.go and output.go: if it shared that
// code it would confirm the same bug rather than catch it. It keeps its own
// struct definitions, its own strict decode, and resolves line and column its
// own way.
package validate

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

var validTypes = map[string]bool{
	"quote": true, "numeric": true, "event": true, "entity": true,
	"definition": true, "causal": true, "other": true,
}

var validConfidence = map[string]bool{"high": true, "medium": true, "low": true}

type position struct {
	Line   int `json:"line"`
	Column int `json:"column"`
}

type citedFact struct {
	ID         int       `json:"id"`
	Fact       string    `json:"fact"`
	Type       string    `json:"type"`
	Confidence string    `json:"confidence"`
	Verbatim   *string   `json:"verbatim"`
	Position   *position `json:"position"`
}

// inferredFact declares verbatim even though the writer omits it, so a stray
// citation in this array is reported as the contract violation it is instead of
// failing the strict decode with an opaque message.
type inferredFact struct {
	ID         int     `json:"id"`
	Fact       string  `json:"fact"`
	Type       string  `json:"type"`
	Confidence string  `json:"confidence"`
	Verbatim   *string `json:"verbatim"`
}

type summary struct {
	Total           int `json:"total"`
	Verbatim        int `json:"verbatim"`
	Inferred        int `json:"inferred"`
	FailedCitations int `json:"failed_citations"`
	Unlocated       int `json:"unlocated"`
}

type document struct {
	Summary       *summary       `json:"summary"`
	VerbatimFacts []citedFact    `json:"verbatim_facts"`
	InferredFacts []inferredFact `json:"inferred_facts"`
}

// fact is the flattened view the shared per-fact rules run over, keeping track
// of which array the entry came from.
type fact struct {
	ID         int
	Fact       string
	Type       string
	Confidence string
	Verbatim   *string
	Position   *position
	Cited      bool
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

	// Position accounting.
	Positioned int // citations carrying a position that was checked
	Unlocated  int // citations the writer could not place in the source

	Problems []string // contract violations; any of these is a failure
	Warnings []string // traceable but imperfect; reported, never fatal
}

// OK reports whether the document honours the contract.
func (r Result) OK() bool { return len(r.Problems) == 0 }

// Check validates raw JSON against the contract. sourceText may be empty, in
// which case the substring and position checks are skipped and only the
// structural rules run.
func Check(raw []byte, sourceText string) (Result, error) {
	res := Result{TypeCounts: map[string]int{}}

	var doc document
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		// A file from before the split decodes as one unknown field, which says
		// nothing useful on its own.
		var flat struct {
			Facts []json.RawMessage `json:"facts"`
		}
		if json.Unmarshal(raw, &flat) == nil && flat.Facts != nil {
			return res, fmt.Errorf("this is the older flat fact document, with one %q array; "+
				"re-run the extractor to produce the current format", "facts")
		}
		return res, fmt.Errorf("not a valid fact document: %w", err)
	}

	// Flatten, remembering the array each entry came from. The two arrays are a
	// presentation of one list, and every rule below the membership checks
	// applies to that list as a whole.
	all := make([]fact, 0, len(doc.VerbatimFacts)+len(doc.InferredFacts))
	for _, f := range doc.VerbatimFacts {
		all = append(all, fact{f.ID, f.Fact, f.Type, f.Confidence, f.Verbatim, f.Position, true})
	}
	for _, f := range doc.InferredFacts {
		all = append(all, fact{f.ID, f.Fact, f.Type, f.Confidence, f.Verbatim, nil, false})
	}
	res.Facts = len(all)

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
	for _, f := range all {
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

		// Membership: the two arrays exist to separate cited claims from
		// uncited ones, so an entry in the wrong one defeats the whole point.
		if !f.Cited {
			if f.Verbatim != nil {
				problem(f, "is in inferred_facts but carries a verbatim span")
				continue
			}
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
			continue
		}
		if f.Verbatim == nil {
			problem(f, "is in verbatim_facts but has no verbatim span")
			continue
		}

		if haystack != "" {
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

		checkPosition(&res, f, sourceText, problem)
	}

	checkIDs(&res, all)
	checkSummary(&res, doc.Summary, all)
	return res, nil
}

// checkPosition confirms the reported line and column really is where the span
// sits. A position nothing verifies is worse than no position at all: it looks
// authoritative while quietly sending the reader to the wrong line.
func checkPosition(res *Result, f fact, sourceText string, problem func(fact, string, ...any)) {
	if f.Position == nil {
		res.Unlocated++
		return
	}
	if f.Position.Line < 1 || f.Position.Column < 1 {
		problem(f, "position is line %d column %d; both are 1-based",
			f.Position.Line, f.Position.Column)
		return
	}
	if sourceText == "" {
		return
	}

	rest, ok := textFrom(sourceText, f.Position.Line, f.Position.Column)
	if !ok {
		problem(f, "position line %d column %d is past the end of the source",
			f.Position.Line, f.Position.Column)
		return
	}
	res.Positioned++

	// Compare the way the span check above does. Verify snapped verbatim to the
	// characters of its own chunk, and a chunk boundary can introduce a line
	// break the source spells as a single space, so squashing both sides is what
	// keeps that from reading as a wrong position.
	want := squash(*f.Verbatim)
	window := []rune(rest)
	if n := len([]rune(want))*3 + 64; len(window) > n {
		window = window[:n]
	}
	if !strings.HasPrefix(strings.ToLower(squash(string(window))), strings.ToLower(want)) {
		problem(f, "position line %d column %d does not point at the verbatim span: %q",
			f.Position.Line, f.Position.Column, truncate(*f.Verbatim, 70))
	}
}

// textFrom returns the source from a 1-based line and column onwards, counting
// columns in runes. Deliberately its own walk rather than a shared helper.
func textFrom(text string, line, column int) (string, bool) {
	runes := []rune(text)
	at, seen := 0, 1
	for seen < line {
		if at >= len(runes) {
			return "", false
		}
		if runes[at] == '\n' {
			seen++
		}
		at++
	}
	at += column - 1
	if at > len(runes) {
		return "", false
	}
	return string(runes[at:]), true
}

// checkIDs holds the ids to the same rule they had when the document was one
// list: 1..N, no gaps, no repeats. Splitting the list must not renumber it, and
// a union that is not 1..N also catches a fact lost or duplicated by the split.
func checkIDs(res *Result, all []fact) {
	ids := make([]int, len(all))
	for i, f := range all {
		ids[i] = f.ID
	}
	sort.Ints(ids)
	for i, id := range ids {
		if id != i+1 {
			res.Problems = append(res.Problems, fmt.Sprintf(
				"ids across both arrays should be 1..%d with no gaps or repeats; found %d where %d was expected",
				len(all), id, i+1))
			break
		}
	}
}

// checkSummary recomputes the header block. A count that drifts from the facts
// beneath it is a quiet lie to whoever reads the top of the file and stops.
func checkSummary(res *Result, s *summary, all []fact) {
	if s == nil {
		res.Problems = append(res.Problems, "summary is missing")
		return
	}
	var want summary
	for _, f := range all {
		if f.Cited {
			want.Verbatim++
			if f.Position == nil {
				want.Unlocated++
			}
			continue
		}
		want.Inferred++
		if f.Confidence != "low" {
			want.FailedCitations++
		}
	}
	want.Total = want.Verbatim + want.Inferred

	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"total", s.Total, want.Total},
		{"verbatim", s.Verbatim, want.Verbatim},
		{"inferred", s.Inferred, want.Inferred},
		{"failed_citations", s.FailedCitations, want.FailedCitations},
		{"unlocated", s.Unlocated, want.Unlocated},
	} {
		if c.got != c.want {
			res.Problems = append(res.Problems,
				fmt.Sprintf("summary.%s is %d but the facts count %d", c.name, c.got, c.want))
		}
	}
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
	fmt.Fprintf(&b, "\nposition: %d checked, %d not located", r.Positioned, r.Unlocated)
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
