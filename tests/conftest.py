import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import sys


# Ensure env vars are set before the app/settings are imported.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("UPLOADS_ROOT", "./test_uploads")
os.environ.setdefault("VECTORSTORE_ROOT", "./test_vectorstore")

# Make `import app...` work when pytest runs from repo root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def client():
    # Clean state once per test run.
    for p in ["./test.db", "./test_uploads", "./test_vectorstore"]:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    from app.main import app

    with TestClient(app) as c:
        yield c

