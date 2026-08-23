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
	raw := []byte(`{"facts":[
		{"id":1,"fact":"Acme Corp is a company.","type":"entity","confidence":"low","verbatim":null},
		{"id":2,"fact":"Acme Corp is profitable.","type":"entity","confidence":"high","verbatim":null},
		{"id":3,"fact":"Acme Corp grew.","type":"event","confidence":"medium","verbatim":null}
	]}`)

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
	raw := []byte(`{"facts":[
		{"id":1,"fact":"Revenue grew.","type":"numeric","confidence":"high",
		 "verbatim":"revenue increased by nearly a fifth"}
	]}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if res.OK() {
		t.Fatal("a verbatim that is not a source span must be a problem")
	}
	if !strings.Contains(res.Problems[0], "not a span of the source") {
		t.Errorf("wrong problem reported: %q", res.Problems[0])
	}
}

// Structural rules: ids, vocabularies, duplicates.
func TestStructuralRules(t *testing.T) {
	raw := []byte(`{"facts":[
		{"id":2,"fact":"Acme Corp reported revenue.","type":"figures","confidence":"certain",
		 "verbatim":"reported €4.2 million in revenue for 2023"},
		{"id":2,"fact":"Acme Corp reported revenue!","type":"numeric","confidence":"high",
		 "verbatim":"reported €4.2 million in revenue for 2023"}
	]}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(res.Problems, "\n")
	for _, want := range []string{
		"id is 2 but should be 1",
		`type "figures" is outside the allowed set`,
		`confidence "certain" is outside the allowed set`,
		"duplicates fact",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing problem %q in:\n%s", want, joined)
		}
	}
}

// An exact copy passes clean; a re-cased span is a warning, not a problem.
func TestExactAndRecased(t *testing.T) {
	raw := []byte(`{"facts":[
		{"id":1,"fact":"Acme Corp reported EUR 4.2 million for 2023.","type":"numeric","confidence":"high",
		 "verbatim":"reported €4.2 million in revenue for 2023"},
		{"id":2,"fact":"Revenue was up 18 percent.","type":"numeric","confidence":"high",
		 "verbatim":"Up 18% from the prior year"}
	]}`)

	res, err := Check(raw, source)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK() {
		t.Fatalf("unexpected problems: %v", res.Problems)
	}
	if res.Exact != 1 || res.Recased != 1 {
		t.Errorf("Exact = %d, Recased = %d, want 1 and 1", res.Exact, res.Recased)
	}
}

// Unknown JSON fields are rejected outright — the strict decode is part of
// being a genuine second opinion.
func TestUnknownFieldsRejected(t *testing.T) {
	raw := []byte(`{"facts":[{"id":1,"fact":"x","type":"other","confidence":"low","verbatim":null,"extra":1}]}`)
	if _, err := Check(raw, source); err == nil {
		t.Fatal("a document with unknown fields must not validate")
	}
}
