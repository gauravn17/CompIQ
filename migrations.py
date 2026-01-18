"""
CompIQ Database Migrations
Alembic-style migrations for database schema management.

Demonstrates:
- Database schema versioning
- Forward and backward migrations
- Migration history tracking
- Safe schema changes
"""
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a single migration."""
    version: str
    description: str
    upgrade_sql: List[str]
    downgrade_sql: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class MigrationManager:
    """
    Manages database migrations.
    
    Similar to Alembic but simplified for SQLite.
    For PostgreSQL, you would use actual Alembic.
    """
    
    MIGRATIONS_TABLE = "_migrations"
    
    def __init__(self, db_path: str = "comparables.db"):
        self.db_path = db_path
        self._migrations: List[Migration] = []
        self._init_migrations_table()
        self._register_migrations()
    
    def _init_migrations_table(self):
        """Create migrations tracking table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                    version TEXT PRIMARY KEY,
                    description TEXT,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _register_migrations(self):
        """Register all migrations in order."""
        
        # Migration 001: Initial schema
        self._migrations.append(Migration(
            version="001",
            description="Initial schema - searches and comparables tables",
            upgrade_sql=[
                """
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_name TEXT NOT NULL,
                    target_data TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    num_comparables INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS comparables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_id INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    validation_score REAL NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY (search_id) REFERENCES searches (id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS companies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    ticker TEXT,
                    exchange TEXT,
                    is_public BOOLEAN,
                    last_verified DATETIME,
                    verification_data TEXT
                )
                """
            ],
            downgrade_sql=[
                "DROP TABLE IF EXISTS companies",
                "DROP TABLE IF EXISTS comparables",
                "DROP TABLE IF EXISTS searches"
            ]
        ))
        
        # Migration 002: Add indexes
        self._migrations.append(Migration(
            version="002",
            description="Add indexes for common queries",
            upgrade_sql=[
                "CREATE INDEX IF NOT EXISTS idx_searches_timestamp ON searches(timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_searches_target ON searches(target_name)",
                "CREATE INDEX IF NOT EXISTS idx_comparables_search ON comparables(search_id)",
                "CREATE INDEX IF NOT EXISTS idx_comparables_ticker ON comparables(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker)"
            ],
            downgrade_sql=[
                "DROP INDEX IF EXISTS idx_searches_timestamp",
                "DROP INDEX IF EXISTS idx_searches_target",
                "DROP INDEX IF EXISTS idx_comparables_search",
                "DROP INDEX IF EXISTS idx_comparables_ticker",
                "DROP INDEX IF EXISTS idx_companies_ticker"
            ]
        ))
        
        # Migration 003: Add ETL tracking
        self._migrations.append(Migration(
            version="003",
            description="Add ETL job tracking table",
            upgrade_sql=[
                """
                CREATE TABLE IF NOT EXISTS etl_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    records_input INTEGER DEFAULT 0,
                    records_processed INTEGER DEFAULT 0,
                    records_failed INTEGER DEFAULT 0,
                    run_hash TEXT,
                    error_message TEXT,
                    started_at DATETIME,
                    completed_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_etl_jobs_status ON etl_jobs(status)",
                "CREATE INDEX IF NOT EXISTS idx_etl_jobs_created ON etl_jobs(created_at DESC)"
            ],
            downgrade_sql=[
                "DROP INDEX IF EXISTS idx_etl_jobs_created",
                "DROP INDEX IF EXISTS idx_etl_jobs_status",
                "DROP TABLE IF EXISTS etl_jobs"
            ]
        ))
        
        # Migration 004: Add financial snapshots
        self._migrations.append(Migration(
            version="004",
            description="Add financial data snapshots for historical tracking",
            upgrade_sql=[
                """
                CREATE TABLE IF NOT EXISTS financial_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    snapshot_date DATE NOT NULL,
                    market_cap REAL,
                    revenue_ttm REAL,
                    ev_to_revenue REAL,
                    revenue_growth REAL,
                    profit_margin REAL,
                    employees INTEGER,
                    raw_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, exchange, snapshot_date)
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON financial_snapshots(ticker, exchange)",
                "CREATE INDEX IF NOT EXISTS idx_snapshots_date ON financial_snapshots(snapshot_date DESC)"
            ],
            downgrade_sql=[
                "DROP INDEX IF EXISTS idx_snapshots_date",
                "DROP INDEX IF EXISTS idx_snapshots_ticker",
                "DROP TABLE IF EXISTS financial_snapshots"
            ]
        ))
        
        # Migration 005: Add user preferences (future feature)
        self._migrations.append(Migration(
            version="005",
            description="Add API keys and rate limiting tables",
            upgrade_sql=[
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT,
                    tier TEXT DEFAULT 'free',
                    rate_limit INTEGER DEFAULT 100,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_used_at DATETIME
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL,
                    window_start DATETIME NOT NULL,
                    request_count INTEGER DEFAULT 0,
                    UNIQUE(key_hash, window_start)
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_rate_limits_key ON rate_limits(key_hash, window_start)"
            ],
            downgrade_sql=[
                "DROP INDEX IF EXISTS idx_rate_limits_key",
                "DROP TABLE IF EXISTS rate_limits",
                "DROP TABLE IF EXISTS api_keys"
            ]
        ))
    
    def get_applied_versions(self) -> List[str]:
        """Get list of applied migration versions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT version FROM {self.MIGRATIONS_TABLE} ORDER BY version"
            )
            return [row[0] for row in cursor.fetchall()]
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied."""
        applied = set(self.get_applied_versions())
        return [m for m in self._migrations if m.version not in applied]
    
    def upgrade(self, target_version: Optional[str] = None) -> List[str]:
        """
        Apply pending migrations.
        
        Args:
            target_version: Optional target version (applies all if None)
        
        Returns:
            List of applied migration versions
        """
        applied = []
        pending = self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        with sqlite3.connect(self.db_path) as conn:
            for migration in pending:
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                
                try:
                    for sql in migration.upgrade_sql:
                        conn.execute(sql)
                    
                    conn.execute(
                        f"INSERT INTO {self.MIGRATIONS_TABLE} (version, description) VALUES (?, ?)",
                        (migration.version, migration.description)
                    )
                    
                    conn.commit()
                    applied.append(migration.version)
                    logger.info(f"Migration {migration.version} applied successfully")
                    
                except Exception as e:
                    logger.error(f"Migration {migration.version} failed: {str(e)}")
                    conn.rollback()
                    raise
        
        return applied
    
    def downgrade(self, target_version: str) -> List[str]:
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Version to rollback to (exclusive)
        
        Returns:
            List of rolled back migration versions
        """
        rolled_back = []
        applied = self.get_applied_versions()
        
        # Get migrations to rollback (in reverse order)
        to_rollback = [
            m for m in reversed(self._migrations)
            if m.version in applied and m.version > target_version
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for migration in to_rollback:
                logger.info(f"Rolling back migration {migration.version}: {migration.description}")
                
                try:
                    for sql in migration.downgrade_sql:
                        conn.execute(sql)
                    
                    conn.execute(
                        f"DELETE FROM {self.MIGRATIONS_TABLE} WHERE version = ?",
                        (migration.version,)
                    )
                    
                    conn.commit()
                    rolled_back.append(migration.version)
                    logger.info(f"Migration {migration.version} rolled back successfully")
                    
                except Exception as e:
                    logger.error(f"Rollback of {migration.version} failed: {str(e)}")
                    conn.rollback()
                    raise
        
        return rolled_back
    
    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = self.get_applied_versions()
        pending = self.get_pending_migrations()
        
        return {
            "database": self.db_path,
            "total_migrations": len(self._migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "current_version": applied[-1] if applied else None,
            "latest_version": self._migrations[-1].version if self._migrations else None
        }
    
    def show_history(self) -> List[Dict[str, Any]]:
        """Show migration history with timestamps."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM {self.MIGRATIONS_TABLE} ORDER BY version"
            )
            return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI for running migrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CompIQ Database Migrations")
    parser.add_argument("command", choices=["status", "upgrade", "downgrade", "history"])
    parser.add_argument("--db", default="comparables.db", help="Database path")
    parser.add_argument("--version", help="Target version for upgrade/downgrade")
    
    args = parser.parse_args()
    
    manager = MigrationManager(args.db)
    
    if args.command == "status":
        status = manager.get_status()
        print("\n" + "=" * 50)
        print("Migration Status")
        print("=" * 50)
        print(f"Database: {status['database']}")
        print(f"Current version: {status['current_version'] or 'None'}")
        print(f"Latest version: {status['latest_version']}")
        print(f"Applied: {status['applied_count']}/{status['total_migrations']}")
        if status['pending_versions']:
            print(f"Pending: {', '.join(status['pending_versions'])}")
        else:
            print("No pending migrations")
    
    elif args.command == "upgrade":
        print(f"Upgrading database: {args.db}")
        applied = manager.upgrade(args.version)
        if applied:
            print(f"Applied migrations: {', '.join(applied)}")
        else:
            print("No migrations to apply")
    
    elif args.command == "downgrade":
        if not args.version:
            print("Error: --version required for downgrade")
            return
        print(f"Downgrading to version: {args.version}")
        rolled_back = manager.downgrade(args.version)
        if rolled_back:
            print(f"Rolled back: {', '.join(rolled_back)}")
        else:
            print("No migrations to rollback")
    
    elif args.command == "history":
        history = manager.show_history()
        print("\n" + "=" * 50)
        print("Migration History")
        print("=" * 50)
        for entry in history:
            print(f"  {entry['version']} | {entry['description']} | {entry['applied_at']}")


if __name__ == "__main__":
    main()
