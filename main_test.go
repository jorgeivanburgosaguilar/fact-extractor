package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// system-instruction.md restates the fact schema in prose, because the JSON
// Schema cannot teach the model what the fields *mean*. That duplication is
// deliberate, but it can drift: adding a field to schemas/facts.json without
// telling the model about it produces output the grammar allows and the
// instruction never explains.
//
// Both copies are embedded in the binary, so this compares what actually ships.
func TestSystemInstructionMatchesSchema(t *testing.T) {
	var schema struct {
		Properties struct {
			Facts struct {
				Items struct {
					Properties map[string]json.RawMessage `json:"properties"`
					Required   []string                   `json:"required"`
				} `json:"items"`
			} `json:"facts"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(defaultSchema, &schema); err != nil {
		t.Fatalf("schemas/facts.json is not valid JSON: %v", err)
	}

	fields := schema.Properties.Facts.Items.Properties
	if len(fields) == 0 {
		t.Fatal("no fact fields found in schemas/facts.json")
	}

	instruction := string(defaultSystemInstruction)
	for name := range fields {
		if !strings.Contains(instruction, `"`+name+`"`) {
			t.Errorf("schema field %q is never mentioned in system-instruction.md: "+
				"the model is not told what it means", name)
		}
	}

	// And the reverse: the required list is the contract the grammar enforces,
	// so every required field must be one the instruction knows about.
	for _, name := range schema.Properties.Facts.Items.Required {
		if _, ok := fields[name]; !ok {
			t.Errorf("schemas/facts.json requires %q but does not define it", name)
		}
	}
}

// The seven categories are a closed vocabulary enforced by the grammar. If the
// instruction and the schema disagree, the model is taught to emit a value the
// grammar will reject.
func TestFactTypeVocabularyMatchesSchema(t *testing.T) {
	var schema struct {
		Properties struct {
			Facts struct {
				Items struct {
					Properties struct {
						Type struct {
							Enum []string `json:"enum"`
						} `json:"type"`
						Confidence struct {
							Enum []string `json:"enum"`
						} `json:"confidence"`
					} `json:"properties"`
				} `json:"items"`
			} `json:"facts"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(defaultSchema, &schema); err != nil {
		t.Fatalf("schemas/facts.json is not valid JSON: %v", err)
	}

	items := schema.Properties.Facts.Items.Properties
	instruction := string(defaultSystemInstruction)

	if len(items.Type.Enum) != 7 {
		t.Errorf("expected 7 fact types in the schema, got %d: %v",
			len(items.Type.Enum), items.Type.Enum)
	}
	for _, v := range items.Type.Enum {
		if !strings.Contains(instruction, "`"+v+"`") {
			t.Errorf("fact type %q is in the schema but never explained in system-instruction.md", v)
		}
	}
	for _, v := range items.Confidence.Enum {
		if !strings.Contains(instruction, "`"+v+"`") {
			t.Errorf("confidence level %q is in the schema but never explained in system-instruction.md", v)
		}
	}
}
