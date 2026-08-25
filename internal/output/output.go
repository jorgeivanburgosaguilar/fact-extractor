// Package output renders a verified fact document into the form written to
// result.json.
//
// The pipeline works on a flat list because that is what the model returns and
// what merging and verification operate on. A person reading the result wants
// something else: the claims backed by a citation separated from the ones that
// are not, and for each citation, where in the source to look. This package is
// that last step and nothing more — it runs after inference is over, adds no
// claim of its own, and never changes a fact.
//
// The model's schema (schemas/facts.json) is deliberately untouched by any of
// this. It stays byte-identical so extractions stay comparable between models
// and between system instructions; position is derived here from the source
// text, never asked of the model.
package output

import (
	"encoding/json"
	"sort"

	"fact-extractor/internal/facts"
)

// Position is where a verbatim span starts in the source, as a person would
// navigate to it. Both are 1-based, and the column counts runes rather than
// bytes so it lines up with what an editor shows.
type Position struct {
	Line   int `json:"line"`
	Column int `json:"column"`
}

// CitedFact is a fact whose verbatim span survived verification, so it can be
// pointed at a place in the source.
type CitedFact struct {
	facts.Fact
	// Position is null when the span could not be located in the whole source;
	// see Build for the one case where that can happen.
	Position *Position `json:"position"`
}

// InferredFact is a fact with no citation. Verbatim is omitted rather than
// written as null: in this array it is null by definition and carries nothing.
//
// Confidence is what still matters here, because null verbatim means two
// different things — "low" is a fact the model honestly inferred, anything
// higher is a citation that failed verification.
type InferredFact struct {
	ID         int    `json:"id"`
	Fact       string `json:"fact"`
	Type       string `json:"type"`
	Confidence string `json:"confidence"`
}

// Summary is the accounting for the document, so the shape of a long result is
// readable without scrolling it.
type Summary struct {
	Total           int `json:"total"`
	Verbatim        int `json:"verbatim"`
	Inferred        int `json:"inferred"`
	FailedCitations int `json:"failed_citations"` // verbatim null above low confidence
	Unlocated       int `json:"unlocated"`        // cited, but not found in the whole source
}

// Document is what result.json holds.
type Document struct {
	Summary       *Summary       `json:"summary"`
	VerbatimFacts []CitedFact    `json:"verbatim_facts"`
	InferredFacts []InferredFact `json:"inferred_facts"`
}

// Build splits a verified document in two and locates every citation in
// sourceText.
//
// Facts are partitioned on whether verbatim is null and on nothing else, so the
// two arrays always partition the input exactly. Ids are carried through
// untouched: they are the fact's identity and stay 1..N across the union of both
// arrays, never renumbered per array.
func Build(doc *facts.Document, sourceText string) *Document {
	out := &Document{
		Summary:       &Summary{},
		VerbatimFacts: []CitedFact{},
		InferredFacts: []InferredFact{},
	}
	if doc == nil {
		return out
	}

	src := facts.NewSource(sourceText)
	lines := lineStarts(sourceText)

	for _, f := range doc.Facts {
		if f.Verbatim == nil {
			out.InferredFacts = append(out.InferredFacts, InferredFact{
				ID:         f.ID,
				Fact:       f.Fact,
				Type:       f.Type,
				Confidence: f.Confidence,
			})
			if f.Confidence != "low" {
				out.Summary.FailedCitations++
			}
			continue
		}

		cited := CitedFact{Fact: f}
		// Verify checked this span against its own chunk; we are searching the
		// whole document. The two can disagree where textsplit packed sentences
		// together with a "\n\n" the source did not have, but Source collapses
		// whitespace runs on both sides of the comparison, so such a span still
		// matches here. If one genuinely does not, the fact keeps its citation
		// and simply reports no position — never dropped, never moved.
		if _, at, ok := src.Locate(*f.Verbatim); ok {
			cited.Position = positionOf(lines, at)
		} else {
			out.Summary.Unlocated++
		}
		out.VerbatimFacts = append(out.VerbatimFacts, cited)
	}

	out.Summary.Verbatim = len(out.VerbatimFacts)
	out.Summary.Inferred = len(out.InferredFacts)
	out.Summary.Total = out.Summary.Verbatim + out.Summary.Inferred
	return out
}

// Encode renders the document as the indented JSON written to disk.
func Encode(d *Document) ([]byte, error) {
	raw, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

// lineStarts records the rune offset each line begins at, so a position lookup
// is a binary search rather than a rescan of the document per fact.
func lineStarts(text string) []int {
	starts := []int{0}
	for i, r := range []rune(text) {
		if r == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// positionOf turns a rune offset into a 1-based line and column.
//
// A \r\n source needs no special handling: the \r sits at the end of a line, so
// it cannot affect a column counted from the following line's start.
func positionOf(starts []int, at int) *Position {
	// The last line beginning at or before the offset. starts[0] is 0 and at is
	// never negative, so this is always in range.
	i := sort.SearchInts(starts, at+1) - 1
	return &Position{Line: i + 1, Column: at - starts[i] + 1}
}
