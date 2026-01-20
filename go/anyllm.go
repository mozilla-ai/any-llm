package anyllm

import (
	"context"
	"fmt"
)

// LLMClient defines the interface for interacting with LLM providers.
type LLMClient interface {
	// Completion performs a chat completion request.
	Completion(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (*ChatCompletion, error)

	// CompletionStream performs a streaming chat completion request.
	// Returns a channel of chunks and a channel for errors.
	CompletionStream(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (<-chan ChatCompletionChunk, <-chan error)

	// ProviderName returns the name of the provider.
	ProviderName() string
}

// ProviderFactory is a function type that creates a new LLMClient.
type ProviderFactory func(config *ClientConfig) (LLMClient, error)

// providerRegistry holds the registered provider factories.
var providerRegistry = make(map[Provider]ProviderFactory)

// RegisterProvider registers a provider factory for the given provider.
func RegisterProvider(provider Provider, factory ProviderFactory) {
	providerRegistry[provider] = factory
}

// Create creates a new LLMClient for the specified provider.
func Create(provider Provider, config *ClientConfig) (LLMClient, error) {
	factory, ok := providerRegistry[provider]
	if !ok {
		return nil, fmt.Errorf("provider '%s' is not supported", provider)
	}

	if config == nil {
		config = &ClientConfig{}
	}

	return factory(config)
}

func init() {
	// Register the OpenAI provider
	RegisterProvider(ProviderOpenAI, NewOpenAIProvider)
}
