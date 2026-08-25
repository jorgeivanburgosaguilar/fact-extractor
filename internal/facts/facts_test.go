package facts

import "testing"

func doc(fs ...Fact) *Document { return &Document{Facts: fs} }

func TestMergeRenumbersSequentially(t *testing.T) {
	a := doc(Fact{ID: 1, Fact: "First."}, Fact{ID: 2, Fact: "Second."})
	b := doc(Fact{ID: 1, Fact: "Third."}, Fact{ID: 2, Fact: "Fourth."})

	got := Merge([]*Document{a, b})
	if len(got.Facts) != 4 {
		t.Fatalf("expected 4 facts, got %d", len(got.Facts))
	}
	for i, f := range got.Facts {
		if f.ID != i+1 {
			t.Errorf("fact %d has id %d, want %d", i, f.ID, i+1)
		}
	}
}

// Chunk boundaries overlap in meaning, so the same fact can be reported twice.
func TestMergeDropsDuplicatesAcrossChunks(t *testing.T) {
	a := doc(Fact{ID: 1, Fact: "Acme was founded in 1984."})
	b := doc(Fact{ID: 1, Fact: "acme was founded in 1984"}, Fact{ID: 2, Fact: "Acme has 300 staff."})

	got := Merge([]*Document{a, b})
	if len(got.Facts) != 2 {
		t.Fatalf("expected the duplicate to be dropped, got %d facts: %+v", len(got.Facts), got.Facts)
	}
	if got.Facts[1].ID != 2 {
		t.Errorf("ids should stay contiguous after a drop, got %d", got.Facts[1].ID)
	}
}

func TestMergeSkipsNilAndEmpty(t *testing.T) {
	got := Merge([]*Document{nil, doc(Fact{ID: 1, Fact: "  "}), doc(Fact{ID: 1, Fact: "Real."})})
	if len(got.Facts) != 1 || got.Facts[0].Fact != "Real." {
		t.Fatalf("expected only the real fact, got %+v", got.Facts)
	}
}

// Merging nothing must still give an empty slice rather than a nil one, so the
// document encodes as an empty array instead of null.
func TestMergeOfNothingIsEmptyNotNil(t *testing.T) {
	got := Merge(nil)
	if got.Facts == nil {
		t.Fatal("expected an empty slice, got nil")
	}
	if len(got.Facts) != 0 {
		t.Fatalf("expected no facts, got %d", len(got.Facts))
	}
}

func TestParseAcceptsNullVerbatim(t *testing.T) {
	d, err := Parse(`{"facts":[{"id":1,"fact":"A.","type":"other","confidence":"low","verbatim":null}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if d.Facts[0].Verbatim != nil {
		t.Errorf("expected a nil verbatim, got %q", *d.Facts[0].Verbatim)
	}
}

func TestParseRejectsNonJSON(t *testing.T) {
	if _, err := Parse("Here are the facts: ..."); err == nil {
		t.Fatal("expected an error for non-JSON output")
	}
}
