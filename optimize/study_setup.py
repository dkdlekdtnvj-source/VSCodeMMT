"""Optuna 스터디 및 스토리지 초기화 유틸리티."""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Dict, List, Optional

import optuna
from sqlalchemy import event
from sqlalchemy.engine import make_url

LOGGER = logging.getLogger("optimize")


def _mask_storage_url(url: str) -> str:
    if not url:
        return ""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return url


def _make_sqlite_storage(
    url: str,
    *,
    timeout_sec: int = 120,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
) -> optuna.storages.RDBStorage:
    """SQLite 스토리지에 WAL 모드와 타임아웃을 적용해 생성합니다."""

    connect_args = {"timeout": timeout_sec, "check_same_thread": False}
    engine_kwargs = {"connect_args": connect_args, "pool_pre_ping": True}
    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval or None,
        grace_period=grace_period or None,
    )

    @event.listens_for(storage.engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:  # type: ignore[unused-ignore]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=60000;")

            wal_pragmas = (
                "PRAGMA journal_mode=WAL;",
                "PRAGMA synchronous=NORMAL;",
                "PRAGMA temp_store=MEMORY;",
            )

            for attempt in range(5):
                try:
                    for pragma in wal_pragmas:
                        cursor.execute(pragma)
                except sqlite3.OperationalError as exc:  # pragma: no cover - 환경 의존
                    is_locked = "database is locked" in str(exc).lower()
                    if is_locked and attempt < 4:
                        time.sleep(0.2 * (attempt + 1))
                        continue

                    LOGGER.warning(
                        "SQLite PRAGMA 설정 중 오류가 발생했습니다 (WAL 미적용 가능성): %s",
                        exc,
                    )
                else:
                    break
        finally:
            cursor.close()

    return storage


def _make_rdb_storage(
    url: str,
    *,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    pool_timeout: Optional[int] = None,
    pool_recycle: Optional[int] = None,
    isolation_level: Optional[str] = None,
    connect_timeout: Optional[int] = None,
    statement_timeout_ms: Optional[int] = None,
) -> optuna.storages.RDBStorage:
    """PostgreSQL 등 외부 RDB 용도의 Optuna 스토리지를 생성합니다."""

    engine_kwargs: Dict[str, object] = {"pool_pre_ping": True}

    if pool_size is not None:
        engine_kwargs["pool_size"] = pool_size
    if max_overflow is not None:
        engine_kwargs["max_overflow"] = max_overflow
    if pool_timeout is not None:
        engine_kwargs["pool_timeout"] = pool_timeout
    if pool_recycle is not None:
        engine_kwargs["pool_recycle"] = pool_recycle
    if isolation_level:
        engine_kwargs["isolation_level"] = isolation_level

    connect_args: Dict[str, object] = {}
    if connect_timeout is not None:
        connect_args["connect_timeout"] = connect_timeout

    options_parts: List[str] = []  # type: ignore[var-annotated]
    if statement_timeout_ms is not None:
        options_parts.append(f"-c statement_timeout={statement_timeout_ms}")

    url_info = None
    try:
        url_info = make_url(url)
    except Exception:
        url_info = None

    is_postgres = bool(url_info and url_info.drivername.startswith("postgresql"))
    if is_postgres:
        engine_kwargs.setdefault("pool_size", 5)
        engine_kwargs.setdefault("max_overflow", 10)
        engine_kwargs.setdefault("pool_recycle", 1800)
        if connect_timeout is None:
            connect_args.setdefault("connect_timeout", 10)
        options_parts.append("-c timezone=UTC")

    if options_parts:
        existing_options = str(connect_args.get("options", "")).strip()
        if existing_options:
            options_parts.insert(0, existing_options)
        connect_args["options"] = " ".join(part for part in options_parts if part)

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval or None,
        grace_period=grace_period or None,
    )

    return storage


__all__ = ["_mask_storage_url", "_make_sqlite_storage", "_make_rdb_storage"]
