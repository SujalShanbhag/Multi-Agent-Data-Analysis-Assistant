import sys
from pathlib import Path

# Ensure root directory is in sys.path for serverless execution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app

# Export ASGI application handler for Vercel serverless functions
handler = app
