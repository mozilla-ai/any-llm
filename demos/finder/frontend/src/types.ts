export interface ProviderStatus {
  name: string;
  display_name: string;
  api_key_configured: boolean;
  env_var: string;
  supports_list_models: boolean;
  missing_packages: boolean;
  error?: string;
}

export interface ModelInfo {
  id: string;
  provider: string;
  provider_display_name: string;
  object?: string;
  created?: number;
  owned_by?: string;
}

export interface SearchResponse {
  query: string;
  models: ModelInfo[];
  total_found: number;
  provider_errors: Array<{
    provider: string;
    error: string;
  }>;
}

export interface AllModelsResponse {
  models: ModelInfo[];
  total_models: number;
  provider_errors: Array<{
    provider: string;
    error: string;
  }>;
}

export interface ProviderStatusResponse {
  providers: ProviderStatus[];
}
