import os
import asyncio
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from groq import Groq
from mem0 import Memory

load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = str(os.getenv("TELEGRAM_ALLOWED_USER_ID", "")).strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PORT = int(os.getenv("PORT", 10000))

CHAT_MODEL = "qwen/qwen3.8-27b"
EXTRACTION_MODEL = "gemini-3.8-flash"

# In-memory sliding window for turn-by-turn dialogue
chat_history = {}

# Load Persona
SOUL_PATH = os.path.join("config", "SOUL.md")
SYSTEM_PROMPT = "You are an elite, proactive AI PersonalThe error occurs because `aiohttp` is not installed in the environment where your application is running on Render.

Here is how to resolve it:

### 1. Add `aiohttp` to your dependencies

Add `aiohttp` to your dependency file in your repository:

* **If using `requirements.txt`:**
  Add this line:
  ```text
  aiohttp