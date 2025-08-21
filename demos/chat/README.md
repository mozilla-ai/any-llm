# any-llm Demo: List Models & Completions

This demo showcases the `list_models` feature of any-llm, allowing you to discover available models from different providers and use them for completions.

## Features

- **Provider Discovery**: View all providers that support the `list_models` API
- **Model Listing**: Dynamically fetch available models from any supported provider
- **Model Filtering**: Search and filter through available models with real-time filtering
- **Streaming Chat**: Real-time streaming responses with character-by-character display
- **Thinking Content**: Collapsible display of model reasoning for supported providers
- **Auto-scrolling**: Chat automatically scrolls to follow the conversation
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

3. Set up provider environment variables:
   ```bash
   # Set API keys for the providers you want to use
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   export MISTRAL_API_KEY="your-mistral-api-key"
   # ... add other provider API keys as needed
   ```

4. Run the server:
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

1. **Set API Keys**: Ensure you have set the appropriate environment variables for the providers you want to use (see Backend setup step 3)
2. **Select a Provider**: Choose from the list of providers that support `list_models`
3. **Load Models**: Models will automatically load when you select a provider
4. **Filter Models**: Use the filter box to search through available models
5. **Select a Model**: Click on a model from the filtered list
6. **Start Chatting**: Send messages to test the completion functionality with real-time streaming responses

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
