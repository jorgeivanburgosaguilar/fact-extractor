// Package facts parses and merges the fact-extraction JSON produced per chunk.
package facts

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
)

// Fact mirrors one entry of the schema in schemas/facts.json.
type Fact struct {
	ID         int     `json:"id"`
	Fact       string  `json:"fact"`
	Type       string  `json:"type"`
	Confidence string  `json:"confidence"`
	Verbatim   *string `json:"verbatim"`
}

// Document is the top-level object the model returns.
type Document struct {
	Facts []Fact `json:"facts"`
}

// Parse decodes one model response. The grammar guarantees valid JSON, so a
// failure here means something upstream went wrong (wrong profile, truncated
// output) and the caller should preserve the raw text.
func Parse(raw string) (*Document, error) {
	var doc Document
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &doc); err != nil {
		return nil, fmt.Errorf("model output is not valid fact JSON: %w", err)
	}
	return &doc, nil
}

// Merge concatenates per-chunk documents into one, dropping duplicates that
// straddle a chunk boundary and renumbering ids 1..N in order of appearance.
func Merge(docs []*Document) *Document {
	merged := &Document{Facts: []Fact{}}
	seen := make(map[string]bool)

	for _, d := range docs {
		if d == nil {
			continue
		}
		for _, f := range d.Facts {
			key := normalize(f.Fact)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			f.ID = len(merged.Facts) + 1
			merged.Facts = append(merged.Facts, f)
		}
	}
	return merged
}

// normalize reduces a fact to a comparison key: lowercase, letters and digits
// only. Two chunks describing the same sentence should collapse to one entry
// even if punctuation or spacing differs.
func normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}
