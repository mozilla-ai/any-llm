# any-llm Model Finder

A demo application that helps you search for AI models across different providers to see where specific models are available. This tool shows you which providers you have configured with API keys and allows you to search for models across all your configured providers.

![Model Finder Demo](./assets/model_finder_demo.gif)

## Features

- **Provider Status Dashboard**: See which providers you have API keys configured for
- **Model Search**: Search for specific models across all configured providers
- **Browse All Models**: View all available models from your configured providers
- **Real-time Results**: Get instant feedback on model availability
- **Provider Error Reporting**: See which providers have issues and why

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

3. Set up provider environment variables for the providers you want to search:
   ```bash
   # Example API keys - set the ones you want to use, or have them already set in your terminal
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   export MISTRAL_API_KEY="your-mistral-api-key"
   export GROQ_API_KEY="your-groq-api-key"
   # ... add other provider API keys as needed
   ```

   The application will automatically detect which providers you have configured and only search those providers. See the [any-llm providers documentation](https://mozilla-ai.github.io/any-llm/providers/) to understand what environment variables are expected for each provider.

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

1. **Check Provider Status**: The sidebar shows all available providers and indicates which ones you have configured with API keys
2. **Search for Models**: Enter a search term (like "gpt-4", "claude", "llama") to find models matching that pattern
3. **Browse All Models**: Click "Browse All Models" to see every model available from your configured providers
4. **View Results**: Models are displayed with their provider, name, and additional metadata when available

## API Endpoints

The backend provides these endpoints:

- `GET /provider-status` - Get the status of all providers (API key configured, supports list_models, etc.)
- `POST /search-models` - Search for models matching a query across configured providers
- `GET /all-models` - Get all models from all configured providers

## Troubleshooting

### No providers configured
If you see "No providers are configured with API keys":
1. Make sure you've set the required environment variables
2. Restart the backend server after setting environment variables
3. Check the provider status panel to see which specific environment variables are needed

### Provider errors
The application will show provider-specific errors in the results. Common issues:
- **API key not configured**: Set the required environment variable
- **Missing packages**: Install additional dependencies with `pip install any-llm-sdk[provider_name]`
- **API errors**: Check your API key validity and provider service status

### Search returns no results
- Try broader search terms (e.g., "gpt" instead of "gpt-4-turbo-specific-version")
- Check that you have providers configured that actually offer the models you're searching for
- Some providers may have rate limiting - wait a moment and try again
