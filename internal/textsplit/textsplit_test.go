package textsplit

import (
	"strings"
	"testing"
)

// words counts whitespace-separated words, a stand-in tokenizer that makes the
// arithmetic in these tests obvious.
func words(s string) (int, error) { return len(strings.Fields(s)), nil }

func TestShortTextStaysWhole(t *testing.T) {
	chunks, err := Split("one two three", 100, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0] != "one two three" {
		t.Fatalf("expected the text unchanged in one chunk, got %q", chunks)
	}
}

func TestSplitsOnParagraphBoundaries(t *testing.T) {
	text := "aa bb cc dd\n\nee ff gg hh\n\nii jj kk ll"
	chunks, err := Split(text, 4, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %q", len(chunks), chunks)
	}
	for i, c := range chunks {
		if n, _ := words(c); n > 4 {
			t.Errorf("chunk %d has %d tokens, over the budget of 4", i, n)
		}
	}
}

func TestPacksSmallParagraphsTogether(t *testing.T) {
	text := "aa bb\n\ncc dd\n\nee ff\n\ngg hh"
	chunks, err := Split(text, 4, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 packed chunks, got %d: %q", len(chunks), chunks)
	}
}

func TestOversizedParagraphFallsBackToSentences(t *testing.T) {
	text := "One two three. Four five six. Seven eight nine."
	chunks, err := Split(text, 3, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected one chunk per sentence, got %d: %q", len(chunks), chunks)
	}
}

func TestOversizedSentenceIsHardSplit(t *testing.T) {
	text := strings.TrimSpace(strings.Repeat("word ", 50))
	chunks, err := Split(text, 10, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected a hard split, got %d chunk(s)", len(chunks))
	}
}

// No content may be lost: every word of the source must survive the split.
func TestSplitPreservesAllWords(t *testing.T) {
	text := "Alpha bravo charlie. Delta echo foxtrot.\n\nGolf hotel india juliet kilo.\n\nLima mike."
	chunks, err := Split(text, 4, words)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Fields(strings.Join(chunks, " "))
	want := strings.Fields(text)
	if len(got) != len(want) {
		t.Fatalf("word count changed: %d in, %d out\n%q", len(want), len(got), chunks)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("word %d changed: %q -> %q", i, want[i], got[i])
		}
	}
}

func TestDecimalsDoNotBreakSentences(t *testing.T) {
	got := sentences("The rate was 3.5 percent in 2024. It fell after that.")
	if len(got) != 2 {
		t.Fatalf("expected 2 sentences, got %d: %q", len(got), got)
	}
}

func TestEmptyInputIsAnError(t *testing.T) {
	if _, err := Split("   \n\n  ", 100, words); err == nil {
		t.Fatal("expected an error for empty input")
	}
}

func TestEstimateIsPositive(t *testing.T) {
	if n := Estimate("hello world"); n <= 0 {
		t.Fatalf("estimate should be positive, got %d", n)
	}
}
