// Package textsplit cuts long documents into chunks that fit comfortably in
// context. Chunking beats raising the context window: prefill attention cost
// grows quadratically, and models reliably drop mid-document facts once the
// prompt gets long.
package textsplit

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Counter returns the token count of a string.
//
// The interface stays open because a real tokenizer is the better measurement,
// but nothing currently supplies one: Ollama exposes no tokenize endpoint on its
// public API, so the CLI passes EstimateCounter. The slack in the context budget
// is what absorbs the difference.
type Counter func(string) (int, error)

// Estimate approximates the token count without a tokenizer: roughly 3.6
// characters per token for prose in Latin scripts. Deliberately pessimistic.
func Estimate(s string) int {
	return utf8.RuneCountInString(s)*10/36 + 1
}

// EstimateCounter adapts Estimate to the Counter signature.
func EstimateCounter(s string) (int, error) { return Estimate(s), nil }

// Split breaks text into chunks of at most maxTokens each, preferring
// paragraph boundaries, then sentence boundaries, and only hard-cutting when a
// single sentence is itself oversized. Returns a single chunk when the whole
// text already fits.
func Split(text string, maxTokens int, count Counter) ([]string, error) {
	if maxTokens <= 0 {
		return nil, fmt.Errorf("maxTokens must be positive, got %d", maxTokens)
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, fmt.Errorf("input text is empty")
	}

	total, err := count(trimmed)
	if err != nil {
		return nil, err
	}
	if total <= maxTokens {
		return []string{trimmed}, nil
	}

	units, err := sizedUnits(paragraphs(trimmed), maxTokens, count)
	if err != nil {
		return nil, err
	}
	return pack(units, maxTokens, "\n\n"), nil
}

// unit is a piece of text with its measured token count.
type unit struct {
	text string
	n    int
}

// sizedUnits measures each paragraph and recursively breaks down any that are
// individually larger than the budget.
func sizedUnits(parts []string, maxTokens int, count Counter) ([]unit, error) {
	var out []unit
	for _, p := range parts {
		n, err := count(p)
		if err != nil {
			return nil, err
		}
		if n <= maxTokens {
			out = append(out, unit{p, n})
			continue
		}
		for _, s := range sentences(p) {
			sn, err := count(s)
			if err != nil {
				return nil, err
			}
			if sn <= maxTokens {
				out = append(out, unit{s, sn})
				continue
			}
			for _, piece := range hardSplit(s, maxTokens, sn) {
				pn, err := count(piece)
				if err != nil {
					return nil, err
				}
				out = append(out, unit{piece, pn})
			}
		}
	}
	return out, nil
}

// pack greedily fills chunks up to the budget.
func pack(units []unit, maxTokens int, sep string) []string {
	var chunks []string
	var cur []string
	curN := 0

	flush := func() {
		if len(cur) > 0 {
			chunks = append(chunks, strings.Join(cur, sep))
			cur, curN = nil, 0
		}
	}
	for _, u := range units {
		if curN > 0 && curN+u.n > maxTokens {
			flush()
		}
		cur = append(cur, u.text)
		curN += u.n
	}
	flush()
	return chunks
}

// paragraphs splits on blank lines, keeping non-empty blocks.
func paragraphs(text string) []string {
	normalized := strings.ReplaceAll(text, "\r\n", "\n")
	var out []string
	for _, block := range strings.Split(normalized, "\n\n") {
		if b := strings.TrimSpace(block); b != "" {
			out = append(out, b)
		}
	}
	if len(out) == 0 {
		out = []string{strings.TrimSpace(normalized)}
	}
	return out
}

// sentences splits on terminal punctuation followed by whitespace. It is a
// heuristic, not a parser: a wrong boundary costs a slightly odd chunk edge,
// never a lost fact, because every chunk is extracted in full.
func sentences(p string) []string {
	var out []string
	var b strings.Builder
	runes := []rune(p)

	for i, r := range runes {
		b.WriteRune(r)
		if r != '.' && r != '!' && r != '?' && r != '\n' {
			continue
		}
		// Only break when whitespace (or end of text) follows, so decimals
		// and abbreviations stay intact.
		if i+1 < len(runes) && !isSpace(runes[i+1]) {
			continue
		}
		if s := strings.TrimSpace(b.String()); s != "" {
			out = append(out, s)
		}
		b.Reset()
	}
	if s := strings.TrimSpace(b.String()); s != "" {
		out = append(out, s)
	}
	if len(out) == 0 {
		out = []string{p}
	}
	return out
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

// hardSplit cuts an oversized sentence on word boundaries, using the measured
// token count to pick a rune budget.
func hardSplit(s string, maxTokens, measured int) []string {
	runes := []rune(s)
	if measured <= 0 {
		measured = 1
	}
	runesPerToken := float64(len(runes)) / float64(measured)
	budget := int(float64(maxTokens) * runesPerToken * 0.9)
	if budget < 200 {
		budget = 200
	}

	var out []string
	for len(runes) > budget {
		cut := budget
		for cut > budget/2 && !isSpace(runes[cut]) {
			cut--
		}
		if cut <= budget/2 {
			cut = budget
		}
		out = append(out, strings.TrimSpace(string(runes[:cut])))
		runes = runes[cut:]
	}
	if rest := strings.TrimSpace(string(runes)); rest != "" {
		out = append(out, rest)
	}
	return out
}
