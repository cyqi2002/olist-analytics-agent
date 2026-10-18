import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "data" / "olist.duckdb"))
MODEL = os.getenv("ANALYTICS_MODEL", "claude-opus-5-5")
LOG_DIR = ROOT / "logs"

ROW_LIMIT = 1000  # LIMIT added to queries that don't have one
QUERY_TIMEOUT_S = 15
MAX_RETRIES = 2
