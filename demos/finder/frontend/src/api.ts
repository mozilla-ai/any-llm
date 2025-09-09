import { ProviderStatusResponse, SearchResponse, AllModelsResponse } from './types';

const API_BASE = 'http://localhost:8000';

export async function fetchProviderStatus(): Promise<ProviderStatusResponse> {
  const response = await fetch(`${API_BASE}/provider-status`);
  if (!response.ok) {
    throw new Error('Failed to fetch provider status');
  }
  return response.json();
}

export async function searchModels(query: string): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search-models`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to search models');
  }

  return response.json();
}

export async function fetchAllModelsStream(
  onStatus: (message: string, progress: number, total: number) => void,
  onProviderComplete: (provider: string, providerDisplay: string, models: any[]) => void,
  onProviderError: (provider: string, providerDisplay: string, error: string) => void,
  onComplete: (models: any[], totalModels: number, providerErrors: any[]) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/all-models`);

    if (!response.ok) {
      throw new Error('Failed to fetch all models');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

        for (const line of lines) {
          if (line.trim() === '') continue; // Skip empty lines

          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();

            if (data === '[DONE]') {
              return;
            }

            if (data) {
              try {
                const parsedChunk = JSON.parse(data);

                if (parsedChunk.type === 'status') {
                  onStatus(parsedChunk.message, parsedChunk.progress, parsedChunk.total);
                } else if (parsedChunk.type === 'provider_complete') {
                  onProviderComplete(parsedChunk.provider, parsedChunk.provider_display, parsedChunk.models);
                } else if (parsedChunk.type === 'provider_error') {
                  onProviderError(parsedChunk.provider, parsedChunk.provider_display, parsedChunk.error);
                } else if (parsedChunk.type === 'complete') {
                  onComplete(parsedChunk.models, parsedChunk.total_models, parsedChunk.provider_errors);
                }
              } catch (e) {
                console.warn('Failed to parse chunk:', data, 'Error:', e);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Streaming failed');
  }
}
