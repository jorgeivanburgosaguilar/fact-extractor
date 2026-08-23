// Package benchmark scores an extraction against a hand-authored gold list.
//
// It answers exactly one question: did a new model, or an edited
// system-instruction.md, make extraction worse? It is not a general evaluation
// workbench, and it deliberately does not try to measure quality in the
// abstract.
//
// The unit of comparison is the source span. Because facts.Verify guarantees
// every verbatim is either null or a real substring of the input, a gold fact
// and an extracted fact can be compared character-for-character with no fuzzy
// matching anywhere in the pipeline.
package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"fact-extractor/internal/facts"
)

// Threshold is how many gold facts may go unmatched before a case fails.
//
// Run-to-run variance is the nature of the model: 17 facts where a previous run
// found 18 means one marginal claim fell the other side of a near-tie, and both
// sets are fully verified. Losing four or more means the model stopped
// extracting, which is a real and otherwise silent failure.
const Threshold = 3

// Gold is one fact a correct extraction must find, anchored to the source.
type Gold struct {
	Verbatim string `json:"verbatim"`
	Type     string `json:"type,omitempty"`
}

// Case pairs a source text with the facts that must come out of it.
type Case struct {
	Source string `json:"source"`
	Gold   []Gold `json:"gold"`

	Name string `json:"-"` // corpus file this came from
}

// LoadCases reads every *.gold.json in dir.
func LoadCases(dir string) ([]Case, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.gold.json"))
	if err != nil {
		return nil, fmt.Errorf("scanning %s: %w", dir, err)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, fmt.Errorf("no *.gold.json files in %s", dir)
	}

	cases := make([]Case, 0, len(paths))
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", p, err)
		}
		var c Case
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", p, err)
		}
		if c.Source == "" {
			return nil, fmt.Errorf("%s: %q is missing", p, "source")
		}
		if len(c.Gold) == 0 {
			return nil, fmt.Errorf("%s: the gold list is empty, so it can never fail", p)
		}
		c.Name = filepath.Base(p)
		// Sources are named relative to the corpus file.
		c.Source = filepath.Join(dir, filepath.Base(c.Source))
		cases = append(cases, c)
	}
	return cases, nil
}

// Result is the score for one case.
type Result struct {
	Name       string
	Extracted  int
	Matched    int
	Missed     []Gold // gold facts nothing accounted for
	Unanchored []Gold // gold spans not found in the source: the gold list is wrong

	// Reported, never scored.
	TypeMismatch int // matched, but classified differently than the gold says
	Inferred     int // verbatim null, confidence low - honest inference
	Fabricated   int // verbatim null, confidence above low - a citation that failed
}

// Pass reports whether the case is within the tolerance.
func (r Result) Pass() bool { return len(r.Missed) <= Threshold && len(r.Unanchored) == 0 }

// Score compares one extraction against its gold list.
//
// doc must already have been through facts.Verify, so its spans are exact.
func Score(name string, doc *facts.Document, sourceText string, gold []Gold) Result {
	src := facts.NewSource(sourceText)
	res := Result{Name: name, Extracted: len(doc.Facts)}

	// Collect the extracted spans. Verify has already snapped them to the source,
	// so each is a real substring and no normalisation is needed here.
	type span struct {
		text string
		idx  int
		used bool // already accounted for a gold fact; see the loop below
	}
	var spans []span
	for i := range doc.Facts {
		f := &doc.Facts[i]
		if f.Verbatim == nil {
			if f.Confidence == "low" {
				res.Inferred++
			} else {
				res.Fabricated++
			}
			continue
		}
		spans = append(spans, span{text: *f.Verbatim, idx: i})
	}

	for _, g := range gold {
		// Resolve the gold span against the real source first. A gold entry that
		// is not in its own source text is an authoring mistake, and silently
		// counting it as a miss would blame the model for it.
		anchor, ok := src.Find(g.Verbatim)
		if !ok {
			res.Unanchored = append(res.Unanchored, g)
			continue
		}

		// A gold fact is found if any extracted span covers the same ground.
		// Equality is too strict: the model may cite a tighter clause than the
		// gold list names, or a wider one that swallows it, and both mean it
		// caught the fact. Containment either way is still exact-character
		// comparison, because both strings are real substrings of the source.
		//
		// Matching is one-to-one. A single wide span - a whole sentence, say -
		// can contain several gold spans, and letting it satisfy all of them
		// inflates the score: an early version reported 18 of 18 gold facts
		// matched from an extraction that held only 15 facts in total. Each
		// extracted span may therefore account for at most one gold fact.
		match := -1
		for i, s := range spans {
			if s.used {
				continue
			}
			if strings.Contains(s.text, anchor) || strings.Contains(anchor, s.text) {
				match = s.idx
				spans[i].used = true
				break
			}
		}
		if match < 0 {
			res.Missed = append(res.Missed, g)
			continue
		}
		res.Matched++
		if g.Type != "" && doc.Facts[match].Type != g.Type {
			res.TypeMismatch++
		}
	}
	return res
}
