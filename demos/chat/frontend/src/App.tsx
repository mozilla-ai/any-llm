import React, { useState, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import { Provider, Model, Message } from './types';
import { fetchProviders, fetchModels, createStreamingCompletion } from './api';
import './App.css';

interface ThinkingBoxProps {
  thinking: string;
}

function ThinkingBox({ thinking }: ThinkingBoxProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="thinking-box">
      <button
        className="thinking-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className="thinking-icon">{isExpanded ? '▼' : '▶'}</span>
        Thinking ({thinking.length} characters)
      </button>
      {isExpanded && (
        <div className="thinking-content">
          <pre>{thinking}</pre>
        </div>
      )}
    </div>
  );
}

function App() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [modelFilter, setModelFilter] = useState<string>('');
  const chatInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadProviders();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const loadModels = async () => {
      if (!selectedProvider) {
        setModels([]);
        setSelectedModel('');
        return;
      }

      setLoading(true);
      setError('');
      try {
        const modelList = await fetchModels(selectedProvider);
        setModels(modelList);
        setSelectedModel('');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models');
        setModels([]);
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, [selectedProvider]);

  const loadProviders = async () => {
    try {
      const providerList = await fetchProviders();
      setProviders(providerList);
    } catch (err) {
      setError('Failed to load providers');
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !selectedProvider || !selectedModel) return;

    const userMessage: Message = { role: 'user', content: inputMessage.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputMessage('');
    setLoading(true);
    setError('');

    const assistantMessage: Message = { role: 'assistant', content: '', thinking: '', model: selectedModel };
    const messagesWithAssistant = [...newMessages, assistantMessage];
    setMessages(messagesWithAssistant);

    try {
              await createStreamingCompletion(
          selectedProvider,
          selectedModel,
          newMessages,
          (chunk) => {
            if (chunk.choices && chunk.choices[0]?.delta) {
              const delta = chunk.choices[0].delta;
              flushSync(() => {
                setMessages(prev => {
                  const updated = [...prev];
                  const lastMessage = updated[updated.length - 1];
                  if (lastMessage.role === 'assistant') {
                    const newMessage: Message = {
                      ...lastMessage,
                      content: delta.content ? lastMessage.content + delta.content : lastMessage.content,
                      thinking: delta.thinking ? (lastMessage.thinking || '') + delta.thinking : lastMessage.thinking
                    };
                    updated[updated.length - 1] = newMessage;
                  }
                  return updated;
                });
              });

              // Force DOM update by directly manipulating the last message element
              setTimeout(() => {
                const messageElements = document.querySelectorAll('.message.assistant');
                if (messageElements.length > 0) {
                  const lastElement = messageElements[messageElements.length - 1];
                  const contentElement = lastElement.querySelector('.message-content');
                  if (contentElement && delta.content) {
                    const currentContent = contentElement.textContent || '';
                    const newContent = currentContent + delta.content;
                    contentElement.textContent = newContent;
                  }
                }
              }, 0);
            }
          },
        () => {
          setLoading(false);
        },
        (error) => {
          setError(error);
          setLoading(false);
        },
        0.7
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setLoading(false);
    }
  };

  const selectModel = (modelId: string) => {
    setSelectedModel(modelId);
    setError('');
    // Focus the chat input after a short delay to ensure the input is rendered
    setTimeout(() => {
      chatInputRef.current?.focus();
    }, 100);
  };

  const clearChat = () => {
    setMessages([]);
    setError('');
  };

  const filteredModels = models.filter(model =>
    model.id.toLowerCase().includes(modelFilter.toLowerCase())
  );

  return (
    <div className="app">
      <div className="header">
        <h1>any-llm Demo</h1>
        <p>Explore models from different providers and test completions</p>
      </div>

      <div className="main-content">
        <div className="sidebar">
          <div className="section">
            <label htmlFor="provider-select">Provider:</label>
            <select
              id="provider-select"
              value={selectedProvider}
              onChange={(e) => {
                setSelectedProvider(e.target.value);
                setModels([]);
                setSelectedModel('');
                setMessages([]);
                setError('');
              }}
            >
              <option value="">Select a provider</option>
              {providers.map((provider) => (
                <option key={provider.name} value={provider.name}>
                  {provider.display_name}
                </option>
              ))}
            </select>
          </div>

          {selectedProvider && loading && (
            <div className="section">
              <div className="loading-text">Loading models...</div>
            </div>
          )}

          {models.length > 0 && (
            <>
              <div className="section">
                <div className="model-filter-container">
                  <input
                    type="text"
                    placeholder="Filter models..."
                    value={modelFilter}
                    onChange={(e) => setModelFilter(e.target.value)}
                    className="model-filter-input"
                  />
                </div>
              </div>
              <div className="section">
                <label>Available Models:</label>
                <div className="model-grid">
                  {filteredModels.map((model) => (
                    <div
                      key={model.id}
                      className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
                      onClick={() => selectModel(model.id)}
                    >
                      <div className="model-name">{model.id}</div>
                      {selectedModel === model.id && (
                        <div className="model-selected-indicator">✓</div>
                      )}
                    </div>
                  ))}
                </div>
                {filteredModels.length === 0 && modelFilter && (
                  <div className="no-models-message">
                    No models match "{modelFilter}"
                  </div>
                )}
              </div>
            </>
          )}

          {selectedModel && (
            <div className="section">
              <button onClick={clearChat} className="secondary">
                Clear Chat
              </button>
            </div>
          )}
        </div>

        <div className="chat-area">
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p>Select a provider and model to start chatting</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-role">
                    {message.role}
                    {message.role === 'assistant' && message.model && (
                      <span className="model-name">({message.model})</span>
                    )}
                  </div>
                  {message.thinking && (
                    <ThinkingBox thinking={message.thinking} />
                  )}
                  <div className="message-content">{message.content}</div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {selectedModel && (
            <div className="input-area">
              <input
                ref={chatInputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder={`Chat with ${selectedModel}`}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !inputMessage.trim()}>
                {loading ? 'Sending...' : 'Send'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
