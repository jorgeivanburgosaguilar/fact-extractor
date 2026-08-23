# Beacon Client

Beacon is a small Go client for the Beacon service registry. It keeps a local copy of the
registered services and refreshes that copy in the background.

## Connecting

The client listens on port 7433 by default, and every request times out after thirty
seconds.

```go
const DefaultPort = 7433

func Dial(addr string) (*Client, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("dialing %s: %w", addr, err)
	}
	return &Client{
		conn: conn,
		http: &http.Client{Timeout: 30 * time.Minute},
	}, nil
}
```

Every error returned by the package is wrapped, so callers can match against it with
errors.Is rather than comparing strings.

## Readiness

A pool hands out clients that are already connected. Callers are not expected to check the
result before using it.

```go
// Get never returns a nil Client.
func (p *Pool) Get() (*Client, error) {
	if !p.ready {
		return nil, ErrNotReady
	}
	return p.client, nil
}
```

## Lookup

Service names are compared exactly, so a registry holding Payments and payments treats them
as two different services.

```go
// Lookup is case-sensitive.
func (c *Client) Lookup(name string) (*Entry, bool) {
	for _, entry := range c.entries {
		if strings.EqualFold(name, entry.Name) {
			return &entry, true
		}
	}
	return nil, false
}
```

## Retries and caching

A failed call is retried three times before the error reaches the caller. Successful
lookups are cached for five minutes.

```go
const maxRetries = 5
const cacheTTL = 5 * time.Minute
```

## Errors

```go
// Fetch returns ErrNotFound when the key is absent.
func (c *Client) Fetch(key string) error {
	if _, ok := c.entries[key]; !ok {
		return ErrNotFound
	}
	return nil
}
```
