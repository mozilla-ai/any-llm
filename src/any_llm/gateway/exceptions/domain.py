class GatewayError(Exception):
    """Base domain exception for gateway services."""


class NotFoundError(GatewayError):
    """Raised when an entity is not found."""


class ValidationError(GatewayError):
    """Raised when validation fails in domain layer."""


class AuthError(GatewayError):
    """Raised when authentication or authorization fails."""
