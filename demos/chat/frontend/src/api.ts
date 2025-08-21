import { Provider, Model, Message, StreamChunk } from './types';

const API_BASE = 'http://localhost:8000';

export async function fetchProviders(): Promise<Provider[]> {
  const response = await fetch(`${API_BASE}/providers`);
  if (!response.ok) {
    throw new Error('Failed to fetch providers');
  }
  const data = await response.json();
  return data.providers;
}

export async function fetchModels(provider: string): Promise<Model[]> {
  const response = await fetch(`${API_BASE}/list-models`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      provider,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch models');
  }

  const data = await response.json();
  return data.models;
}

export async function createStreamingCompletion(
  provider: string,
  model: string,
  messages: Message[],
  onChunk: (chunk: StreamChunk) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  temperature?: number
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        provider,
        model,
        messages,
        temperature,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create streaming completion');
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
              onComplete();
              return;
            }

            if (data) {
              try {
                const parsedChunk = JSON.parse(data);
                if (parsedChunk.error) {
                  onError(parsedChunk.error);
                  return;
                }
                onChunk(parsedChunk as StreamChunk);
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
