"""Configuration settings for the Streamlit frontend."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env configuration is loaded before reading environment variables
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path if _env_path.exists() else None)

# Base URL for FastAPI backend (configurable via environment variable)
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Default request timeout in seconds
REQUEST_TIMEOUT: float = float(os.getenv("FRONTEND_REQUEST_TIMEOUT", "30.0"))

# Application metadata
APP_TITLE: str = "Regulatory Affairs Assistant"
APP_SUBTITLE: str = "AI-Powered Decision-Support Assistant for Life Sciences & Regulatory Affairs"
REGULATORY_DISCLAIMER: str = (
    "**Regulatory Notice**: This application is an informational decision-support tool "
    "for life sciences and regulatory affairs professionals. It assists in retrieving internal "
    "regulatory documentation, analyzing guidance, and referencing official drug labeling data. "
    "It does NOT provide medical advice, diagnosis, or treatment recommendations, and does NOT "
    "autonomously approve, reject, or make definitive regulatory decisions. All regulatory filings "
    "and determinations remain the sole responsibility of qualified human professionals."
)
