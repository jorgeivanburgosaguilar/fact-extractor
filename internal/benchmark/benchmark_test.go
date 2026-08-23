package benchmark

import (
	"os"
	"path/filepath"
	"testing"

	"fact-extractor/internal/facts"
)

const corpusDir = "../../corpus"

// Every gold span must be findable in its own source. A gold list that has
// drifted from its text would report misses the model is not responsible for,
// which is worse than having no benchmark at all.
func TestCorpusGoldSpansAnchor(t *testing.T) {
	cases, err := LoadCases(corpusDir)
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range cases {
		raw, err := os.ReadFile(c.Source)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		src := facts.NewSource(string(raw))

		for _, g := range c.Gold {
			if _, ok := src.Find(g.Verbatim); !ok {
				t.Errorf("%s: gold span is not in %s: %q",
					c.Name, filepath.Base(c.Source), g.Verbatim)
			}
		}
	}
}

// A perfect extraction must score as a pass, and the reported counts must line
// up with the gold list.
func TestScorePerfectExtraction(t *testing.T) {
	source := "Acme Corp reported €4.2 million in revenue for 2023, up 18% from the prior year."
	span := "reported €4.2 million in revenue for 2023"

	doc := &facts.Document{Facts: []facts.Fact{
		{ID: 1, Fact: "a", Type: "numeric", Confidence: "high", Verbatim: &span},
	}}
	gold := []Gold{{Verbatim: "Acme Corp reported €4.2 million in revenue for 2023", Type: "numeric"}}

	r := Score("t", doc, source, gold)
	if !r.Pass() {
		t.Errorf("a covered gold fact must pass: %+v", r)
	}
	if r.Matched != 1 {
		t.Errorf("matched = %d, want 1 (a tighter cited span still covers the gold fact)", r.Matched)
	}
}

// Losing more than the tolerance is a failure, and the missed facts are named.
func TestScoreFailsBeyondThreshold(t *testing.T) {
	source := "One two three four five six seven eight."
	doc := &facts.Document{}
	gold := []Gold{
		{Verbatim: "One"}, {Verbatim: "two"}, {Verbatim: "three"}, {Verbatim: "four"},
	}

	r := Score("t", doc, source, gold)
	if r.Pass() {
		t.Error("four unmatched gold facts must fail")
	}
	if len(r.Missed) != 4 {
		t.Errorf("missed = %d, want 4", len(r.Missed))
	}
}

// A fabricated citation must be counted separately from an honest inference.
// After Verify these both carry a null span, and confidence is what tells them
// apart.
func TestScoreSeparatesFabricationFromInference(t *testing.T) {
	source := "Acme Corp is based in Berlin."
	doc := &facts.Document{Facts: []facts.Fact{
		{ID: 1, Fact: "inferred", Confidence: "low", Verbatim: nil},
		{ID: 2, Fact: "fabricated", Confidence: "high", Verbatim: nil},
	}}

	r := Score("t", doc, source, []Gold{{Verbatim: "Acme Corp is based in Berlin"}})
	if r.Inferred != 1 {
		t.Errorf("inferred = %d, want 1", r.Inferred)
	}
	if r.Fabricated != 1 {
		t.Errorf("fabricated = %d, want 1", r.Fabricated)
	}
}

// A gold span that is not in its own source is an authoring error, and must be
// reported as such rather than counted against the model.
func TestScoreFlagsUnanchoredGold(t *testing.T) {
	r := Score("t", &facts.Document{}, "Some source text.", []Gold{{Verbatim: "not present here"}})
	if len(r.Unanchored) != 1 {
		t.Fatalf("unanchored = %d, want 1", len(r.Unanchored))
	}
	if len(r.Missed) != 0 {
		t.Error("an unanchored gold span must not also be counted as missed")
	}
	if r.Pass() {
		t.Error("a case with a broken gold list must not pass")
	}
}

// One extracted span must not satisfy several gold facts. A wide citation - a
// whole sentence - can contain many gold spans, and counting it once per gold
// fact reports more matches than there are facts in the document.
func TestScoreMatchingIsOneToOne(t *testing.T) {
	source := "Acme Corp was founded in 1984 in Bremerhaven and employs 3,200 people."
	wide := "founded in 1984 in Bremerhaven"

	doc := &facts.Document{Facts: []facts.Fact{
		{ID: 1, Fact: "one wide citation", Confidence: "high", Verbatim: &wide},
	}}
	gold := []Gold{
		{Verbatim: "founded in 1984"},
		{Verbatim: "in Bremerhaven"},
	}

	r := Score("t", doc, source, gold)
	if r.Matched != 1 {
		t.Errorf("matched = %d, want 1: a single extracted span cannot account for two gold facts", r.Matched)
	}
	if len(r.Missed) != 1 {
		t.Errorf("missed = %d, want 1", len(r.Missed))
	}
}
