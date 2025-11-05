try:
    from any_llm._version import version as __version__
except ImportError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
