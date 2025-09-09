import React, { useState, useEffect, useMemo } from 'react';
import { ProviderStatus, ModelInfo } from './types';
import { fetchProviderStatus, fetchAllModelsStream } from './api';
import './App.css';

function App() {
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [allModels, setAllModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingStatus, setLoadingStatus] = useState<string>('');
  const [loadingProgress, setLoadingProgress] = useState<{current: number, total: number}>({current: 0, total: 0});
  const [error, setError] = useState<string>('');
  const [providerErrors, setProviderErrors] = useState<Array<{provider: string, error: string}>>([]);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loadedFromCache, setLoadedFromCache] = useState<boolean>(false);

  // Cache management
  const CACHE_KEY = 'any-llm-finder-models';
  const CACHE_TIMESTAMP_KEY = 'any-llm-finder-timestamp';
  const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

  const loadFromCache = (): { models: ModelInfo[], timestamp: Date } | null => {
    try {
      const cachedModels = localStorage.getItem(CACHE_KEY);
      const cachedTimestamp = localStorage.getItem(CACHE_TIMESTAMP_KEY);

      if (cachedModels && cachedTimestamp) {
        const timestamp = new Date(cachedTimestamp);
        const now = new Date();

        // Check if cache is still valid (within CACHE_DURATION)
        if (now.getTime() - timestamp.getTime() < CACHE_DURATION) {
          return {
            models: JSON.parse(cachedModels),
            timestamp
          };
        }
      }
    } catch (error) {
      console.warn('Failed to load from cache:', error);
    }
    return null;
  };

  const saveToCache = (models: ModelInfo[], timestamp: Date) => {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(models));
      localStorage.setItem(CACHE_TIMESTAMP_KEY, timestamp.toISOString());
    } catch (error) {
      console.warn('Failed to save to cache:', error);
    }
  };

  useEffect(() => {
    loadProviderStatus();

    // Try to load from cache first
    const cached = loadFromCache();
    if (cached) {
      setAllModels(cached.models);
      setLastUpdated(cached.timestamp);
      setLoadedFromCache(true);
      setLoading(false);
      setLoadingStatus(`Loaded ${cached.models.length} models from cache`);
    } else {
      setLoadedFromCache(false);
      loadAllModels();
    }
  }, []);

  const loadProviderStatus = async () => {
    try {
      const response = await fetchProviderStatus();
      setProviders(response.providers);
    } catch (err) {
      setError('Failed to load provider status');
    }
  };

  const loadAllModels = async () => {
    setLoading(true);
    setError('');
    setAllModels([]);
    setProviderErrors([]);
    setLoadedFromCache(false);

    await fetchAllModelsStream(
      // onStatus
      (message: string, progress: number, total: number) => {
        setLoadingStatus(message);
        setLoadingProgress({current: progress, total});
      },
      // onProviderComplete
      (provider: string, providerDisplay: string, models: any[]) => {
        setAllModels(prev => [...prev, ...models]);
        setLoadingStatus(`Loaded ${models.length} models from ${providerDisplay}`);
      },
      // onProviderError
      (provider: string, providerDisplay: string, error: string) => {
        setProviderErrors(prev => [...prev, {provider, error}]);
        setLoadingStatus(`Error loading ${providerDisplay}: ${error}`);
      },
      // onComplete
      (models: any[], totalModels: number, errors: any[]) => {
        const timestamp = new Date();
        setAllModels(models);
        setProviderErrors(errors);
        setLastUpdated(timestamp);
        setLoading(false);
        setLoadingStatus(`Loaded ${totalModels} models from all providers`);

        // Save to cache
        saveToCache(models, timestamp);
      },
      // onError
      (error: string) => {
        setError(error);
        setLoading(false);
        setLoadingStatus('');
      }
    );
  };

  // Filter models based on search query
  const filteredModels = useMemo(() => {
    if (!searchQuery.trim()) {
      return allModels;
    }

    const query = searchQuery.toLowerCase().trim();
    return allModels.filter(model =>
      model.id.toLowerCase().includes(query) ||
      model.provider_display_name.toLowerCase().includes(query) ||
      (model.owned_by && model.owned_by.toLowerCase().includes(query))
    );
  }, [allModels, searchQuery]);

  const getStatusIndicator = (provider: ProviderStatus) => {
    if (provider.missing_packages) {
      return <span className="status-indicator status-error">Missing Packages</span>;
    }
    if (!provider.supports_list_models) {
      return <span className="status-indicator status-no-models">No List Models</span>;
    }
    if (provider.api_key_configured) {
      return <span className="status-indicator status-configured">Configured</span>;
    }
    return <span className="status-indicator status-missing">No API Key</span>;
  };

  const configuredProviders = providers.filter(p => p.api_key_configured && p.supports_list_models && !p.missing_packages);

  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  return (
    <div className="app">
      <div className="header">
        <h1>any-llm Model Finder</h1>
        <p>Find AI models across different providers</p>
      </div>

      <div className="main-content">
        <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <div className="sidebar-header">
            <button
              className="sidebar-toggle"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {sidebarCollapsed ? '▶' : '◀'}
            </button>
            {!sidebarCollapsed && <span className="sidebar-title">Controls</span>}
          </div>

          {!sidebarCollapsed && (
            <>
              <div className="section">
                {loading && (
                  <div className="loading-progress">
                    <div className="progress-text">{loadingStatus}</div>
                    {loadingProgress.total > 0 && (
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{width: `${(loadingProgress.current / loadingProgress.total) * 100}%`}}
                        />
                      </div>
                    )}
                  </div>
                )}
                <button
                  className="browse-button"
                  onClick={loadAllModels}
                  disabled={loading}
                >
                  {loading ? 'Loading...' : 'Refresh Models'}
                </button>
                {lastUpdated && !loading && (
                  <div className="last-updated">
                    <div className="last-updated-time">
                      Last updated: {lastUpdated.toLocaleString()}
                    </div>
                    {loadedFromCache && (
                      <div className="cache-status">
                        📋 Loaded from cache
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="section">
                <h3>Provider Status ({configuredProviders.length} configured)</h3>
                <div className="provider-status">
                  {providers.map((provider) => (
                    <div key={provider.name} className="provider-item">
                      <div className="provider-name">{provider.display_name}</div>
                      <div className="provider-status-indicators">
                        {getStatusIndicator(provider)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {configuredProviders.length === 0 && !loading && (
                <div className="section">
                  <div className="error-message">
                    No providers are configured with API keys. Set environment variables and restart the backend to enable model searching.
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="main-area">
          <div className="search-section">
            <div className="search-container">
              <input
                type="text"
                className="search-input"
                placeholder="Filter models (e.g., gpt-4, claude, llama)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={loading}
              />
            </div>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {loading ? (
            <div className="loading-container">
              <div className="loading-header">
                <h3>Loading Models</h3>
                {loadingProgress.total > 0 && (
                  <div className="loading-progress-info">
                    {loadingProgress.current} of {loadingProgress.total} providers completed
                  </div>
                )}
              </div>

              {loadingProgress.total > 0 && (
                <div className="loading-progress-bar">
                  <div
                    className="loading-progress-fill"
                    style={{width: `${(loadingProgress.current / loadingProgress.total) * 100}%`}}
                  />
                </div>
              )}

              <div className="loading-status">
                {loadingStatus || 'Initializing...'}
              </div>

              {/* Show models as they load */}
              {allModels.length > 0 && (
                <div className="loading-preview">
                  <div className="loading-preview-header">
                    <h4>Models Found So Far ({allModels.length})</h4>
                  </div>
                  <div className="loading-model-grid">
                    {allModels.slice(0, 12).map((model, index) => (
                      <div key={`loading-${model.provider}-${model.id}-${index}`} className="loading-model-card">
                        <div className="loading-model-name">{model.id}</div>
                        <div className="loading-model-provider">{model.provider_display_name}</div>
                      </div>
                    ))}
                    {allModels.length > 12 && (
                      <div className="loading-model-card loading-more">
                        <div className="loading-more-text">+{allModels.length - 12} more</div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <>
              {filteredModels.length > 0 && (
                <>
                  <div className="results-header">
                    <h3>
                      {searchQuery.trim() ? `Filtered Results for "${searchQuery}"` : 'All Available Models'}
                    </h3>
                    <div className="results-count">
                      {filteredModels.length} of {allModels.length} model{allModels.length !== 1 ? 's' : ''}
                      {searchQuery.trim() && ' matching filter'}
                    </div>
                  </div>

                  <div className="model-grid">
                    {filteredModels.map((model, index) => (
                      <div key={`${model.provider}-${model.id}-${index}`} className="model-card">
                        <div className="model-name">{model.id}</div>
                        <div className="model-provider">{model.provider_display_name}</div>
                        <div className="model-details">
                          {model.owned_by && <div>Owner: {model.owned_by}</div>}
                          {model.object && <div>Type: {model.object}</div>}
                          {model.created && (
                            <div>Created: {new Date(model.created * 1000).toLocaleDateString()}</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {!loading && filteredModels.length === 0 && !error && (
                <div className="empty-state">
                  {allModels.length === 0
                    ? configuredProviders.length === 0
                      ? 'No providers configured. Set up API keys to browse models.'
                      : 'No models loaded yet. Check provider status or refresh.'
                    : searchQuery.trim()
                      ? `No models found matching "${searchQuery}"`
                      : 'No models available'
                  }
                </div>
              )}

              {providerErrors.length > 0 && (
                <div className="provider-errors">
                  <h4>Provider Errors:</h4>
                  {providerErrors.map((error, index) => (
                    <div key={index} className="provider-error">
                      <strong>{error.provider}:</strong> {error.error}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
