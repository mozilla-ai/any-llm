from any_llm.gateway.core import database as _database

get_db = _database.get_db
init_db = _database.init_db
reset_db = _database.reset_db


def __getattr__(name: str):
    if name in {"_engine", "_SessionLocal"}:
        return getattr(_database, name)
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["get_db", "init_db", "reset_db"]
