export interface Provider {
  name: string;
  display_name: string;
}

export interface Model {
  id: string;
  object: string;
  created?: number;
  owned_by?: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  thinking?: string;
  model?: string;
}

export interface StreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: {
      content?: string;
      thinking?: string;
    };
    finish_reason: string | null;
  }[];
}
