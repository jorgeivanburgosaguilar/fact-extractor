// Command checkfacts validates a result.json against the contract in
// system-instruction.md, independently of the tool that produced it. The
// checking itself lives in internal/validate; this is the file-reading,
// printing, exit-code shell around it.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"fact-extractor/internal/validate"
)

func main() {
	source := flag.String("source", "", "source text the facts were extracted from")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: checkfacts [--source FILE] RESULT.JSON")
		os.Exit(2)
	}
	jsonPath := flag.Arg(0)

	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	var sourceText string
	if *source != "" {
		s, err := os.ReadFile(*source)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(2)
		}
		sourceText = string(s)
	}

	res, err := validate.Check(raw, sourceText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %s: %v\n", jsonPath, err)
		os.Exit(2)
	}

	fmt.Printf("%s\n  %s\n", jsonPath, strings.ReplaceAll(res.Summary(), "\n", "\n  "))
	if len(res.Warnings) > 0 {
		fmt.Printf("  WARN - %d span(s) imperfect but not contract violations:\n", len(res.Warnings))
		for _, w := range res.Warnings {
			fmt.Printf("    %s\n", w)
		}
	}
	if !res.OK() {
		fmt.Printf("  FAIL - %d problem(s):\n", len(res.Problems))
		for _, p := range res.Problems {
			fmt.Printf("    %s\n", p)
		}
		os.Exit(1)
	}
	fmt.Printf("  PASS - schema, ids, vocabularies and verbatim spans all valid\n")
}
