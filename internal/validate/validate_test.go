package validate

import (
	"strings"
	"testing"
)

const source = "Acme Corp reported €4.2 million in revenue for 2023, up 18% from the prior year."

// The two null-verbatim states must be told apart by confidence, and neither
// is a contract violation. This is the matrix the tool's whole confidence
// decision rests on:
//
//	null + low       -> honest inference
//	null + high/med  -> a citation that failed verification (warning, counted)
func TestNullVerbatimConfidenceMatrix(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":3,"verbatim":0,"inferred":3,"failed_citations":2,"unlocated":0},
		"verbatim_facts": [],
		"inferred_facts": [
			{"id":1,"fact":"Acme Corp is a company.","type":"entity","confidence":"low"},
			{"id":2,"fact":"Acme Corp is profitable.","type":"entity","confidence":"high"},
			{"id":3,"fact":"Acme Corp grew.","type":"event","confidence":"medium"}
		]
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK() {
		t.Fatalf("null verbatim must never be a contract violation, got problems: %v", res.Problems)
	}
	if res.NullInferred != 1 {
		t.Errorf("NullInferred = %d, want 1", res.NullInferred)
	}
	if res.FailedCitations != 2 {
		t.Errorf("FailedCitations = %d, want 2", res.FailedCitations)
	}
	if len(res.Warnings) != 2 {
		t.Errorf("failed citations must be warned about, got %d warnings", len(res.Warnings))
	}
}

// A paraphrase is the failure that matters: it breaks traceability.
func TestParaphraseIsAProblem(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":1,"verbatim":1,"inferred":0,"failed_citations":0,"unlocated":1},
		"verbatim_facts": [
			{"id":1,"fact":"Revenue grew.","type":"numeric","confidence":"high",
			 "verbatim":"revenue increased by nearly a fifth","position":null}
		],
		"inferred_facts": []
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if res.OK() {
		t.Fatal("a verbatim that is not a source span must be a problem")
	}
	if !strings.Contains(strings.Join(res.Problems, "\n"), "not a span of the source") {
		t.Errorf("wrong problem reported: %q", res.Problems)
	}
}

// Structural rules: ids, vocabularies, duplicates.
func TestStructuralRules(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":2,"verbatim":2,"inferred":0,"failed_citations":0,"unlocated":0},
		"verbatim_facts": [
			{"id":2,"fact":"Acme Corp reported revenue.","type":"figures","confidence":"certain",
			 "verbatim":"reported €4.2 million in revenue for 2023","position":{"line":1,"column":11}},
			{"id":2,"fact":"Acme Corp reported revenue!","type":"numeric","confidence":"high",
			 "verbatim":"reported €4.2 million in revenue for 2023","position":{"line":1,"column":11}}
		],
		"inferred_facts": []
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(res.Problems, "\n")
	for _, want := range []string{
		`type "figures" is outside the allowed set`,
		`confidence "certain" is outside the allowed set`,
		"duplicates fact",
		"no gaps or repeats",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing problem %q in:\n%s", want, joined)
		}
	}
}

// An exact copy passes clean; a re-cased span is a warning, not a problem.
func TestExactAndRecased(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":2,"verbatim":2,"inferred":0,"failed_citations":0,"unlocated":0},
		"verbatim_facts": [
			{"id":1,"fact":"Acme Corp reported EUR 4.2 million for 2023.","type":"numeric","confidence":"high",
			 "verbatim":"reported €4.2 million in revenue for 2023","position":{"line":1,"column":11}},
			{"id":2,"fact":"Revenue was up 18 percent.","type":"numeric","confidence":"high",
			 "verbatim":"Up 18% from the prior year","position":{"line":1,"column":54}}
		],
		"inferred_facts": []
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.OK() {
		t.Fatalf("unexpected problems: %v", res.Problems)
	}
	if res.Exact != 1 || res.Recased != 1 {
		t.Errorf("Exact = %d, Recased = %d, want 1 and 1", res.Exact, res.Recased)
	}
	if res.Positioned != 2 {
		t.Errorf("Positioned = %d, want 2", res.Positioned)
	}
}

// Unknown JSON fields are rejected outright — the strict decode is part of
// being a genuine second opinion.
func TestUnknownFieldsRejected(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":1,"verbatim":0,"inferred":1,"failed_citations":0,"unlocated":0},
		"verbatim_facts": [],
		"inferred_facts": [{"id":1,"fact":"x","type":"other","confidence":"low","extra":1}]
	}`)
	if _, err := Check(raw, source); err == nil {
		t.Fatal("a document with unknown fields must not validate")
	}
}

// The arrays are the whole point of the split, so an entry in the wrong one is
// a contract violation rather than a cosmetic complaint.
func TestFactsMustBeInTheRightArray(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":2,"verbatim":1,"inferred":1,"failed_citations":0,"unlocated":1},
		"verbatim_facts": [
			{"id":1,"fact":"Acme Corp exists.","type":"entity","confidence":"low","verbatim":null,"position":null}
		],
		"inferred_facts": [
			{"id":2,"fact":"Acme Corp reported revenue.","type":"numeric","confidence":"low",
			 "verbatim":"reported €4.2 million in revenue for 2023"}
		]
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(res.Problems, "\n")
	for _, want := range []string{
		"is in verbatim_facts but has no verbatim span",
		"is in inferred_facts but carries a verbatim span",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing problem %q in:\n%s", want, joined)
		}
	}
}

// A position that looks authoritative and points at the wrong line is worse
// than no position at all, so it must not pass quietly.
func TestWrongPositionIsAProblem(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":1,"verbatim":1,"inferred":0,"failed_citations":0,"unlocated":0},
		"verbatim_facts": [
			{"id":1,"fact":"Revenue was up 18 percent.","type":"numeric","confidence":"high",
			 "verbatim":"up 18% from the prior year","position":{"line":1,"column":1}}
		],
		"inferred_facts": []
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if res.OK() {
		t.Fatal("a position that does not point at the span must be a problem")
	}
	if !strings.Contains(strings.Join(res.Problems, "\n"), "does not point at the verbatim span") {
		t.Errorf("wrong problem reported: %q", res.Problems)
	}
}

// The header block is the part a reader trusts without checking, so it has to
// agree with the facts underneath it.
func TestSummaryMustMatchTheFacts(t *testing.T) {
	raw := []byte(`{
		"summary": {"total":9,"verbatim":0,"inferred":1,"failed_citations":0,"unlocated":0},
		"verbatim_facts": [],
		"inferred_facts": [{"id":1,"fact":"Acme Corp exists.","type":"entity","confidence":"low"}]
	}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(res.Problems, "\n"), "summary.total is 9 but the facts count 1") {
		t.Errorf("a drifted summary must be reported, got: %v", res.Problems)
	}
}
