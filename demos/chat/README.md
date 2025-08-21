# any-llm Demo: List Models & Completions

This demo showcases the `list_models` feature of any-llm, allowing you to discover available models from different providers and use them for completions.

## Features

- **Provider Discovery**: View all providers that support the `list_models` API
- **Model Listing**: Dynamically fetch available models from any supported provider
- **Interactive Chat**: Test completions with selected models
- **Clean UI**: Simple, responsive interface for easy exploration

## Setup

### Backend (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

3. Run the server:
   ```bash
   uv run python main.py
   ```

The API will be available at `http://localhost:8000`

### Frontend (React)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Select a Provider**: Choose from the list of providers that support `list_models`
2. **Configure API Access**: Enter API key and/or base URL if required by the provider
3. **Load Models**: Click "Load Models" to fetch available models from the provider
4. **Select a Model**: Choose a model from the loaded list
5. **Start Chatting**: Send messages to test the completion functionality

## Supported Providers

The demo works with any provider that implements the `list_models` functionality, including:
- OpenAI
- Anthropic
- Google
- Mistral
- Cohere
- Groq
- And many more...

## API Endpoints

- `GET /providers` - List all providers supporting `list_models`
- `POST /list-models` - Get available models for a provider
- `POST /completion` - Create a completion using the selected model
