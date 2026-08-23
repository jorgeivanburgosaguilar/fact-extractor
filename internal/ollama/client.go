// Package ollama is the HTTP client for the Ollama service.
//
// This package only talks to a service; starting and stopping one is
// internal/service's job. Every request carries keep_alive: 0 so the model is
// unloaded the moment the run finishes, which is what keeps the old promise
// that a run leaves no VRAM held.
package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Message is one entry in the chat transcript.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Client talks to one Ollama service.
type Client struct {
	host  string
	model string
	http  *http.Client
}

// New returns a client for the service at host. The timeout is generous: a
// single chunk of a dense document can take minutes to generate on a 6 GB card.
func New(host, model string) *Client {
	return &Client{
		host:  strings.TrimRight(host, "/"),
		model: model,
		http:  &http.Client{Timeout: 30 * time.Minute},
	}
}

type chatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	Stream    bool            `json:"stream"`
	KeepAlive int             `json:"keep_alive"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   map[string]any  `json:"options,omitempty"`
}

type chatResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	EvalDuration    int64  `json:"eval_duration"` // nanoseconds
	Error           string `json:"error"`
}

// Result is one completed generation.
type Result struct {
	Content    string
	PromptN    int
	PredictedN int
	PerSecond  float64
	DoneReason string
}

// Truncated reports whether generation stopped because it ran out of room
// rather than because the model finished. The output is incomplete and the
// caller must not treat it as a whole answer.
func (r Result) Truncated() bool { return r.DoneReason == "length" }

// Chat sends one request and waits for the whole reply.
//
// format carries the JSON Schema that constrains the output. Ollama forwards it
// to llama.cpp as json_schema, so the reply is grammar-constrained rather than
// merely requested politely.
func (c *Client) Chat(msgs []Message, opts map[string]any, format json.RawMessage, keepAlive int) (*Result, error) {
	body, err := json.Marshal(chatRequest{
		Model:     c.model,
		Messages:  msgs,
		Stream:    false,
		KeepAlive: keepAlive,
		Format:    format,
		Options:   opts,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding chat request: %w", err)
	}

	resp, err := c.http.Post(c.host+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("calling %s/api/chat: %w\n%s", c.host, err, hint(c.host))
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading reply from %s: %w", c.host, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s/api/chat returned %s: %s",
			c.host, resp.Status, strings.TrimSpace(string(raw)))
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parsing reply from %s: %w", c.host, err)
	}
	if out.Error != "" {
		return nil, fmt.Errorf("ollama: %s", out.Error)
	}

	res := &Result{
		Content:    out.Message.Content,
		PromptN:    out.PromptEvalCount,
		PredictedN: out.EvalCount,
		DoneReason: out.DoneReason,
	}
	if out.EvalDuration > 0 {
		res.PerSecond = float64(out.EvalCount) / (float64(out.EvalDuration) / 1e9)
	}
	return res, nil
}

// Model reports which model this client sends requests for, so a run can name
// it without the caller having to repeat how it was chosen.
func (c *Client) Model() string { return c.model }

// Model describes one model the service holds.
type Model struct {
	Name string `json:"name"`
}

// Placement reports how a loaded model is split between GPU and CPU. This is
// what `ollama ps` prints in its PROCESSOR column.
//
// It matters because partial CPU offload costs roughly half the speed and
// reports no error anywhere. On the previous runtime the condition was
// undetectable; here it is one HTTP call.
type Placement struct {
	Name       string
	Size       int64 // total bytes the model occupies
	SizeVRAM   int64 // of which, bytes resident on the GPU
	GPUPercent int
}

// FullyOnGPU reports whether the whole model is resident in VRAM.
func (p Placement) FullyOnGPU() bool { return p.Size > 0 && p.SizeVRAM == p.Size }

// String renders the split the way `ollama ps` does.
func (p Placement) String() string {
	if p.Size == 0 {
		return "no model loaded"
	}
	if p.FullyOnGPU() {
		return "100% GPU"
	}
	return fmt.Sprintf("%d%%/%d%% CPU/GPU", 100-p.GPUPercent, p.GPUPercent)
}

// Running returns the placement of the currently loaded model, if any. A zero
// Placement with no error means nothing is loaded.
func (c *Client) Running() (Placement, error) {
	var p Placement

	resp, err := c.http.Get(c.host + "/api/ps")
	if err != nil {
		return p, fmt.Errorf("querying %s/api/ps: %w", c.host, err)
	}
	defer resp.Body.Close()

	var out struct {
		Models []struct {
			Name     string `json:"name"`
			Size     int64  `json:"size"`
			SizeVRAM int64  `json:"size_vram"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return p, fmt.Errorf("reading %s/api/ps: %w", c.host, err)
	}
	if len(out.Models) == 0 {
		return p, nil
	}

	m := out.Models[0]
	p = Placement{Name: m.Name, Size: m.Size, SizeVRAM: m.SizeVRAM}
	if m.Size > 0 {
		p.GPUPercent = int(m.SizeVRAM * 100 / m.Size)
	}
	return p, nil
}

// Warmup loads the model and holds it for keepAlive seconds, so its placement
// can be read before any real work is measured against it.
func (c *Client) Warmup(opts map[string]any, keepAlive int) error {
	_, err := c.Chat(
		[]Message{{Role: "user", Content: "ok"}},
		opts, nil, keepAlive)
	return err
}

// Unload releases the model immediately.
func (c *Client) Unload() error {
	_, err := c.Chat([]Message{{Role: "user", Content: "ok"}}, nil, nil, 0)
	return err
}

// Preflight checks that the service is reachable and that the configured model
// exists, so a run fails in the first second with an actionable message instead
// of somewhere deeper with an opaque one.
func (c *Client) Preflight() error {
	resp, err := c.http.Get(c.host + "/api/tags")
	if err != nil {
		return fmt.Errorf("cannot reach Ollama at %s: %w\n%s", c.host, err, hint(c.host))
	}
	defer resp.Body.Close()

	var list struct {
		Models []Model `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return fmt.Errorf("reading the model list from %s: %w", c.host, err)
	}

	names := make([]string, 0, len(list.Models))
	for _, m := range list.Models {
		// Ollama reports "name" and "name:latest" for the same model.
		if m.Name == c.model || strings.TrimSuffix(m.Name, ":latest") == c.model {
			return nil
		}
		names = append(names, m.Name)
	}

	return fmt.Errorf("Ollama is running at %s but has no model named %q.\n"+
		"Available: %s\n"+
		"Create it with:  ollama create %s -f Modelfile",
		c.host, c.model, strings.Join(names, ", "), c.model)
}

// hint spells out the two things that are almost always wrong on a first run.
func hint(host string) string {
	return "Start the service with 'ollama serve', then check these are set for it\n" +
		"(the VRAM budget in hardware-finetune.md depends on them):\n" +
		"  OLLAMA_FLASH_ATTENTION=1\n" +
		"  OLLAMA_KV_CACHE_TYPE=q8_0\n" +
		"  OLLAMA_NUM_PARALLEL=1\n" +
		"Expected service address: " + host
}
