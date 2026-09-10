package benchmark

import (
	"encoding/json"
	"testing"
)

// NewRun's totals must equal the sum of the per-case numbers it was built
// from, and the overall Pass must go false the moment any one case does.
// Never assert an exact fact count here - these are structural checks on the
// record, not a prediction of what any model will extract.
func TestNewRunTotalsSumCases(t *testing.T) {
	cases := []CaseRecord{
		{Name: "a", Pass: true, Gold: 5, Matched: 5, Missed: 0, Extracted: 5, Fabricated: 0},
		{Name: "b", Pass: false, Gold: 7, Matched: 3, Missed: 4, Extracted: 4, Fabricated: 1},
	}

	run := NewRun("fact-extractor", "max", 1, 1000, 16384, 0, cases)

	if run.Pass {
		t.Error("Pass must be false when any case fails")
	}
	if run.Totals.Cases != 2 {
		t.Errorf("Totals.Cases = %d, want 2", run.Totals.Cases)
	}
	if run.Totals.Passed != 1 {
		t.Errorf("Totals.Passed = %d, want 1", run.Totals.Passed)
	}
	if want := cases[0].Gold + cases[1].Gold; run.Totals.Gold != want {
		t.Errorf("Totals.Gold = %d, want %d", run.Totals.Gold, want)
	}
	if want := cases[0].Matched + cases[1].Matched; run.Totals.Matched != want {
		t.Errorf("Totals.Matched = %d, want %d", run.Totals.Matched, want)
	}
	if want := cases[0].Missed + cases[1].Missed; run.Totals.Missed != want {
		t.Errorf("Totals.Missed = %d, want %d", run.Totals.Missed, want)
	}
	if want := cases[0].Extracted + cases[1].Extracted; run.Totals.Extracted != want {
		t.Errorf("Totals.Extracted = %d, want %d", run.Totals.Extracted, want)
	}
	if want := cases[0].Fabricated + cases[1].Fabricated; run.Totals.Fabricated != want {
		t.Errorf("Totals.Fabricated = %d, want %d", run.Totals.Fabricated, want)
	}
}

// A model with no thinking mode must serialize think as JSON null, not as
// false - the two mean different things (no such setting vs. explicitly off).
func TestRunThinkNilEncodesAsNull(t *testing.T) {
	run := NewRun("fact-extractor-qwen", nil, 1, 1000, 16384, 0, nil)

	enc, err := Encode(run)
	if err != nil {
		t.Fatal(err)
	}

	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(enc, &decoded); err != nil {
		t.Fatal(err)
	}
	if string(decoded["think"]) != "null" {
		t.Errorf(`"think" = %s, want null`, decoded["think"])
	}
}

// think may be an effort-level string as well as a bool - both are valid
// values Ollama accepts, and the record must carry either through untouched.
func TestRunThinkStringLevelEncodesAsString(t *testing.T) {
	run := NewRun("fact-extractor", "max", 1, 1000, 16384, 0, nil)

	enc, err := Encode(run)
	if err != nil {
		t.Fatal(err)
	}

	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(enc, &decoded); err != nil {
		t.Fatal(err)
	}
	if string(decoded["think"]) != `"max"` {
		t.Errorf(`"think" = %s, want "max"`, decoded["think"])
	}
}

// An empty run (no cases) must still round-trip: zero cases is a valid,
// if useless, Run rather than a nil-slice panic.
func TestNewRunEmptyCases(t *testing.T) {
	run := NewRun("m", nil, 1, 1000, 16384, 0, nil)
	if !run.Pass {
		t.Error("a run with zero cases has nothing to fail on, so Pass must be true")
	}
	if run.Totals.Cases != 0 {
		t.Errorf("Totals.Cases = %d, want 0", run.Totals.Cases)
	}
	if _, err := Encode(run); err != nil {
		t.Fatalf("Encode: %v", err)
	}
}
