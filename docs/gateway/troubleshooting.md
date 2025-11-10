# Troubleshooting

## Database connection errors

Make sure the database URL is correct and the database is accessible:

```bash
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://user:pass@host/db'); print('OK')"
```

## Common Issues

### Authentication Errors

- Ensure you're using the correct master key format: `Bearer your-secure-master-key`
- Check that the `X-AnyLLM-Key` header is properly set
- Verify that virtual API keys are active and not expired

### Configuration Issues

- Verify your `config.yml` file is properly formatted
- Check that environment variables are set correctly
- Ensure provider API keys are valid and have proper permissions

### Budget Enforcement

- Check that budgets are properly assigned to users
- Verify budget limits are set correctly
- Monitor user spending to ensure limits are being enforced

## Getting Help

- Check the logs for detailed error messages
- Verify your configuration matches the examples in the documentation
- Ensure all required environment variables are set
