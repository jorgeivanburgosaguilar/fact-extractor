package facts

import "testing"

const src = `Nordwind Energie announced on 14 March 2024 that it will build a 480 MW
offshore wind farm. "This is the largest single investment," said Ingrid
Halvorsen. Twelve of the 74 stations recorded net accretion.`

func find(t *testing.T, span string) (string, bool) {
	t.Helper()
	return NewSource(src).Find(span)
}

func TestFindExactSpan(t *testing.T) {
	got, ok := find(t, "Nordwind Energie announced on 14 March 2024")
	if !ok || got != "Nordwind Energie announced on 14 March 2024" {
		t.Fatalf("ok=%v got=%q", ok, got)
	}
}

// The model routinely lowercases the first letter to fit its sentence.
func TestFindRepairsCase(t *testing.T) {
	got, ok := find(t, "twelve of the 74 stations")
	if !ok {
		t.Fatal("expected a case-insensitive match")
	}
	if got != "Twelve of the 74 stations" {
		t.Fatalf("expected the span snapped to the source casing, got %q", got)
	}
}

// Straight vs curly vs single quotes are the same content.
func TestFindRepairsQuoteCharacters(t *testing.T) {
	got, ok := find(t, "'This is the largest single investment,'")
	if !ok {
		t.Fatal("expected a match despite the quote characters")
	}
	if got != `"This is the largest single investment,"` {
		t.Fatalf("expected the source quote characters back, got %q", got)
	}
}

// A full stop where the source has a different character must not break
// traceability. Here the source continues "... 2024 that it will build".
func TestFindRepairsAddedFullStop(t *testing.T) {
	got, ok := find(t, "Nordwind Energie announced on 14 March 2024.")
	if !ok || got != "Nordwind Energie announced on 14 March 2024" {
		t.Fatalf("ok=%v got=%q", ok, got)
	}
}

// Punctuation that really is part of the span must be kept, not trimmed away.
func TestFindKeepsGenuineTrailingPunctuation(t *testing.T) {
	got, ok := find(t, "offshore wind farm.")
	if !ok || got != "offshore wind farm." {
		t.Fatalf("ok=%v got=%q", ok, got)
	}
}

// A span crossing a newline in the source still matches.
func TestFindAcrossLineBreak(t *testing.T) {
	if _, ok := find(t, "will build a 480 MW offshore wind farm"); !ok {
		t.Fatal("expected the span to match across the line break")
	}
}

// The failure that matters: the model altered a word. There is no honest
// repair, so it must not match.
func TestFindRejectsAlteredWord(t *testing.T) {
	if got, ok := find(t, "Eleven of the 74 stations recorded net accretion"); ok {
		t.Fatalf("expected no match for a corrupted span, got %q", got)
	}
}

func TestFindRejectsAbsentText(t *testing.T) {
	if _, ok := find(t, "a sentence that is simply not there"); ok {
		t.Fatal("expected no match for absent text")
	}
}

func TestVerifyRepairsDropsAndCounts(t *testing.T) {
	exact := "Nordwind Energie announced on 14 March 2024"
	recased := "twelve of the 74 stations"
	bogus := "Eleven of the 74 stations recorded net accretion"

	doc := &Document{Facts: []Fact{
		{ID: 1, Fact: "a", Confidence: "high", Verbatim: &exact},
		{ID: 2, Fact: "b", Confidence: "high", Verbatim: &recased},
		{ID: 3, Fact: "c", Confidence: "high", Verbatim: &bogus},
		{ID: 4, Fact: "d", Confidence: "low", Verbatim: nil},
	}}

	res := Verify(doc, src)
	if res.Exact != 1 || res.Repaired != 1 || res.Dropped != 1 {
		t.Fatalf("counts wrong: %+v", res)
	}
	if *doc.Facts[1].Verbatim != "Twelve of the 74 stations" {
		t.Errorf("fact 2 not snapped to source: %q", *doc.Facts[1].Verbatim)
	}
	if doc.Facts[2].Verbatim != nil {
		t.Errorf("fact 3 should have been dropped, got %q", *doc.Facts[2].Verbatim)
	}
	// Verify must not touch confidence. Forcing a dropped span to "low" would
	// make a citation that failed verification indistinguishable from a fact the
	// model honestly inferred, which is the one thing reading result.json has to
	// be able to tell apart.
	if doc.Facts[2].Confidence != "high" {
		t.Errorf("a dropped span must keep the model's own confidence, got %q",
			doc.Facts[2].Confidence)
	}
	if doc.Facts[3].Verbatim != nil {
		t.Error("an already-null verbatim must stay null")
	}
}

// After Verify, every non-nil span must be a real substring of the source.
func TestVerifyGuaranteesSubstrings(t *testing.T) {
	spans := []string{
		"nordwind energie announced",
		"'This is the largest single investment,' said Ingrid Halvorsen.",
		"Eleven of the 74 stations",
		"offshore wind farm.",
	}
	doc := &Document{}
	for i := range spans {
		doc.Facts = append(doc.Facts, Fact{ID: i + 1, Fact: "x", Confidence: "high", Verbatim: &spans[i]})
	}
	Verify(doc, src)

	for _, f := range doc.Facts {
		if f.Verbatim == nil {
			continue
		}
		if !contains(src, *f.Verbatim) {
			t.Errorf("fact %d verbatim is not a substring of the source: %q", f.ID, *f.Verbatim)
		}
	}
}

func contains(haystack, needle string) bool {
	return len(needle) > 0 && indexOf(haystack, needle) >= 0
}

func indexOf(h, n string) int {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return i
		}
	}
	return -1
}
