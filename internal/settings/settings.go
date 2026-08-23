// Package settings loads settings.json, the one file that describes how this
// tool talks to Ollama.
//
// The file is produced at build time by build.ps1 and is only ever read here.
// There are no profiles, no merging onto compiled-in defaults and no partial
// files: one job needs one flat description of itself, and a value that is
// missing is a mistake worth reporting rather than papering over.
package settings

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Ollama locates the service and the model inside it.
//
// CoderModel is the alternative the --coder flag selects. It exists so two
// models can be scored on the same corpus, and for nothing else: a third model
// or a flag that also swapped the instruction would be the profile system this
// project deliberately removed.
type Ollama struct {
	Host       string `json:"host"`
	Model      string `json:"model"`
	CoderModel string `json:"coder_model"`
	KeepAlive  int    `json:"keep_alive"`
}

// Options is the block sent to Ollama with every request. It is an open map so
// any parameter Ollama understands can be added to settings.json without
// touching this code; unknown keys are forwarded verbatim.
type Options map[string]any

// Service describes whether and how the CLI manages the Ollama service itself.
//
// When Manage is true and no service answers, the CLI starts `Command serve`
// with Env applied and stops it on exit — the whole tuning story then lives in
// this file. When the block is absent, Manage defaults to false: the old
// behaviour, a service the user runs themselves.
type Service struct {
	Manage                bool              `json:"manage"`
	Command               string            `json:"command"`
	StartupTimeoutSeconds int               `json:"startup_timeout_seconds"`
	Env                   map[string]string `json:"env"`
}

// StartupTimeout returns the configured timeout with a sane floor.
func (s Service) StartupTimeout() time.Duration {
	if s.StartupTimeoutSeconds <= 0 {
		return 60 * time.Second
	}
	return time.Duration(s.StartupTimeoutSeconds) * time.Second
}

// Settings is the whole document.
type Settings struct {
	Ollama      Ollama  `json:"ollama"`
	Service     Service `json:"service"`
	SourceModel string  `json:"source_model"`
	ChunkTokens int     `json:"chunk_tokens"`
	Options     Options `json:"options"`

	baseDir string
}

// BaseDir is the directory settings.json was found in. Relative paths resolve
// against it.
func (s *Settings) BaseDir() string { return s.baseDir }

// Resolve turns a possibly-relative path into an absolute one.
func (s *Settings) Resolve(p string) string {
	if p == "" {
		return ""
	}
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Join(s.baseDir, p)
}

// NumCtx reports the context window the options block asks for. The value is
// always sent explicitly, because Ollama silently shrinks an automatic context
// after an out-of-memory retry and an explicit one it cannot touch.
func (s *Settings) NumCtx() int {
	if n, ok := s.Options["num_ctx"].(float64); ok {
		return int(n)
	}
	return 0
}

// Load reads settings.json from dir.
func Load(dir string) (*Settings, error) {
	path := filepath.Join(dir, "settings.json")

	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("settings.json not found at %s\n"+
			"It is produced by build.ps1 and ships beside the executable. "+
			"Run build.ps1, or copy settings.json next to fact-extractor.exe.", path)
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}

	var s Settings
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	s.baseDir = dir

	if err := s.validate(path); err != nil {
		return nil, err
	}
	return &s, nil
}

func (s *Settings) validate(path string) error {
	missing := func(field string) error {
		return fmt.Errorf("%s: %q is missing or empty", path, field)
	}
	if s.Ollama.Host == "" {
		return missing("ollama.host")
	}
	if s.Ollama.Model == "" {
		return missing("ollama.model")
	}
	if s.ChunkTokens <= 0 {
		return fmt.Errorf("%s: \"chunk_tokens\" must be a positive number, got %d",
			path, s.ChunkTokens)
	}
	if len(s.Options) == 0 {
		return missing("options")
	}
	// A managed service needs a launchable command; "ollama" is the only
	// sensible default and an empty string is far more likely an oversight
	// than an intention.
	if s.Service.Manage && s.Service.Command == "" {
		s.Service.Command = "ollama"
	}
	// num_ctx is the one option the tool reasons about itself, so its absence is
	// a real error rather than something to fall back on.
	if s.NumCtx() <= 0 {
		return fmt.Errorf("%s: \"options.num_ctx\" must be a positive number.\n"+
			"It is always sent explicitly, because Ollama reduces an automatic "+
			"context after an out-of-memory retry without saying so.", path)
	}
	return nil
}
