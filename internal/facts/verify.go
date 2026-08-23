package facts

import (
	"strings"
	"unicode"
)

// Source indexes a chunk of input text so extracted spans can be checked
// against it and snapped back to the exact original characters.
//
// The whole value of a fact extractor is that every claim is traceable to the
// text it came from, and a 7B model gets that slightly wrong often enough to
// matter: it re-cases a leading capital, swaps " for ', adds a full stop the
// source does not have, and occasionally alters a word outright. The first
// three are cosmetic and recoverable; the last destroys the citation. We hold
// the source, so we do not have to guess which happened.
type Source struct {
	orig []rune
	norm []rune
	idx  []int // norm[i] came from orig[idx[i]]
}

// NewSource builds the index. Normalisation collapses whitespace runs, folds
// case, and canonicalises quotation marks and dashes, so a span that differs
// from the source only in those respects still matches.
func NewSource(text string) *Source {
	s := &Source{orig: []rune(text)}
	lastSpace := false

	for i, r := range s.orig {
		if unicode.IsSpace(r) {
			// A run of whitespace becomes exactly one space, anchored at the
			// first character of the run.
			if lastSpace || len(s.norm) == 0 {
				continue
			}
			s.norm = append(s.norm, ' ')
			s.idx = append(s.idx, i)
			lastSpace = true
			continue
		}
		lastSpace = false
		s.norm = append(s.norm, canonical(r))
		s.idx = append(s.idx, i)
	}
	// A trailing space would never take part in a useful match.
	if n := len(s.norm); n > 0 && s.norm[n-1] == ' ' {
		s.norm = s.norm[:n-1]
		s.idx = s.idx[:n-1]
	}
	return s
}

// Find locates span in the source and returns the exact original text at that
// position. The returned string is guaranteed to be a substring of the input,
// even when the model's own wording of the span was not.
func (s *Source) Find(span string) (string, bool) {
	q := normalizeSpan(span)
	// Try the span as given first, so punctuation that genuinely belongs to it
	// survives. Only if that fails do we retry without the edges, which is what
	// rescues a span carrying an added full stop or a stray wrapping quote.
	if got, ok := s.search(q); ok {
		return got, true
	}
	return s.search(trimEdges(q))
}

func (s *Source) search(q []rune) (string, bool) {
	if len(q) == 0 || len(q) > len(s.norm) {
		return "", false
	}
	for i := 0; i+len(q) <= len(s.norm); i++ {
		if matchAt(s.norm, q, i) {
			start := s.idx[i]
			end := s.idx[i+len(q)-1]
			return strings.TrimSpace(string(s.orig[start : end+1])), true
		}
	}
	return "", false
}

// VerifyResult counts what Verify had to do, for reporting to the user.
type VerifyResult struct {
	Exact    int // span was already a perfect copy
	Repaired int // span was traceable but not copied exactly; snapped to source
	Dropped  int // span was not in the source at all; set to null
}

// Any reports whether anything needed changing.
func (v VerifyResult) Any() bool { return v.Repaired > 0 || v.Dropped > 0 }

// Verify checks every verbatim span in doc against the text it was extracted
// from. Spans that are traceable are replaced with the exact source characters;
// spans that are not present at all are set to null. Enforcing that here means
// result.json can be trusted: verbatim is always either null or a genuine
// substring of the input.
//
// Confidence is deliberately left alone. An earlier version forced dropped
// spans to "low", which made a citation that failed verification look identical
// to a fact the model honestly inferred. Keeping the model's own value means the
// two read differently in the output:
//
//	verbatim set             -> verified
//	verbatim null, low       -> inferred from the text
//	verbatim null, high/med  -> a citation that failed verification
func Verify(doc *Document, sourceText string) VerifyResult {
	src := NewSource(sourceText)
	var res VerifyResult

	for i := range doc.Facts {
		f := &doc.Facts[i]
		if f.Verbatim == nil {
			continue
		}
		exact, ok := src.Find(*f.Verbatim)
		switch {
		case !ok:
			f.Verbatim = nil
			res.Dropped++
		case exact == *f.Verbatim:
			res.Exact++
		default:
			f.Verbatim = &exact
			res.Repaired++
		}
	}
	return res
}

func matchAt(hay, needle []rune, at int) bool {
	for j, r := range needle {
		if hay[at+j] != r {
			return false
		}
	}
	return true
}

func normalizeSpan(s string) []rune {
	var out []rune
	lastSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if lastSpace || len(out) == 0 {
				continue
			}
			out = append(out, ' ')
			lastSpace = true
			continue
		}
		lastSpace = false
		out = append(out, canonical(r))
	}
	for len(out) > 0 && out[len(out)-1] == ' ' {
		out = out[:len(out)-1]
	}
	return out
}

// canonical folds case and maps the quotation marks and dashes that models
// substitute freely onto one representative each. Both sides of a comparison
// go through it, so collapsing ' onto " cannot create a false match that the
// source text would not also produce.
func canonical(r rune) rune {
	switch r {
	case '\'', '"', '‘', '’', '‚', '“', '”', '„', '«', '»', '`', '´':
		return '"'
	case '–', '—', '−':
		return '-'
	}
	return unicode.ToLower(r)
}

// trimEdges drops leading and trailing punctuation, so a span carrying an
// added full stop or a wrapping quotation mark still finds its source.
func trimEdges(r []rune) []rune {
	isEdge := func(c rune) bool {
		return c == ' ' || c == '"' || c == '.' || c == ',' || c == ';' ||
			c == ':' || c == '!' || c == '?' || c == '(' || c == ')' ||
			c == '[' || c == ']' || c == '-'
	}
	for len(r) > 0 && isEdge(r[0]) {
		r = r[1:]
	}
	for len(r) > 0 && isEdge(r[len(r)-1]) {
		r = r[:len(r)-1]
	}
	return r
}
