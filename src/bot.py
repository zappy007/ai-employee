import os
import time
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
# Suppress noisy HTTP logs that leak access tokens
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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

# In-memory sliding window for turn-by-turn dialogue
chat_history = {}

# Load Persona
SOUL_PATH = os.path.join("config", "SOUL.md")
SYSTEM_PROMPT = "You are an elite, proactive AI Personal Employee."
if os.path.exists(SOUL_PATH):
    with open(SOUL_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Cloud-based Mem0 configuration
EMBEDDING_DIMS = 768

mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "collection_name": "ai_employee_memory_v3",
            "embedding_model_dims": EMBEDDING_DIMS,
        },
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "qwen/qwen3.8-27b",
            "api_key": GROQ_API_KEY,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "embedding_dims": EMBEDDING_DIMS,
            "api_key": GEMINI_API_KEY,
        },
    },
}

memory = Memory.from_config(mem0_config)


def is_authorized(update: Update) -> bool:
    user_id = str(update.effective_user.id).strip()
    return user_id == ALLOWED_USER_ID


def parse_memories(raw_output) -> list:
    items = []
    if isinstance(raw_output, list):
        for entry in raw_output:
            if isinstance(entry, dict) and entry.get("memory"):
                items.append(entry["memory"])
            elif isinstance(entry, str):
                items.append(entry)
    elif isinstance(raw_output, dict):
        for entry in raw_output.get("results", []):
            if isinstance(entry, dict) and entry.get("memory"):
                items.append(entry["memory"])
    return items


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await update.message.reply_text(
        "AI Personal Employee operational on Render Cloud. How can I assist you today?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return

    user_text = update.message.text.strip()
    user_id = str(update.effective_user.id).strip()

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # 1. TWO-TIER RETRIEVAL
    all_facts = []

    # Tier 1: Full persistent profile
    try:
        profile_res = memory.get_all(filters={"user_id": user_id})
        all_facts.extend(parse_memories(profile_res))
    except Exception as e:
        logger.warning(f"Profile retrieval: {e}")

    # Tier 2: Episodic search
    try:
        search_res = memory.search(user_text, filters={"user_id": user_id})
        all_facts.extend(parse_memories(search_res))
    except Exception as e:
        logger.warning(f"Episodic search: {e}")

    unique_memories = list(dict.fromkeys(all_facts))

    retrieved_context = ""
    if unique_memories:
        retrieved_context = "\n- " + "\n- ".join(unique_memories)

    # 2. PROMPT AUGMENTATION
    augmented_system = SYSTEM_PROMPT
    if retrieved_context:
        augmented_system += (
            f"\n\n### KNOWN FACTS & LONG-TERM MEMORY ABOUT THE USER:\n"
            f"{retrieved_context}\n\n"
            f"Directive: Use the facts above as ground-truth context about your employer."
        )

    # 3. SLIDING CONVERSATION WINDOW
    if user_id not in chat_history:
        chat_history[user_id] = []

    messages_payload = [{"role": "system", "content": augmented_system}]
    messages_payload.extend(chat_history[user_id][-8:])
    messages_payload.append({"role": "user", "content": user_text})

    # 4. INFERENCE VIA GROQ
    try:
        completion = groq_client.chat.completions.create(
            messages=messages_payload,
            model=CHAT_MODEL,
        )
        reply_text = completion.choices[0].message.content.strip()
    except Exception as e:
        reply_text = f"Processing error: {e}"

    chat_history[user_id].append({"role": "user", "content": user_text})
    chat_history[user_id].append({"role": "assistant", "content": reply_text})

    # 5. SEND REPLY
    await update.message.reply_text(reply_text)

    # 6. ASYNC EXTRACTION WITH AUTO-RETRY
    def save_memory_task():
        if len(user_text.split()) < 3:
            return
        for attempt in range(3):
            try:
                memory.add(user_text, user_id=user_id)
                logger.info("[Memory Engine] Logged to Qdrant Cloud successfully.")
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"[Memory Engine] Extraction notice: {e}")
                    break

    asyncio.get_running_loop().run_in_executor(None, save_memory_task)


# Lightweight HTTP Health Check Server using standard library
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is healthy and running.")

    def log_message(self, format, *args):
        pass


def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthCheckHandler)
    server.serve_forever()


def main():
    if not TELEGRAM_BOT_TOKEN or not ALLOWED_USER_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_ID")

    # Start HTTP server daemon for Render uptime port binding
    http_thread = threading.Thread(target=run_health_server, daemon=True)
    http_thread.start()
    print(f"Health server successfully bound to port {PORT}")

    print(f"Starting AI Employee for User ID: {ALLOWED_USER_ID}...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is live! Listening for Telegram messages...")
    app.run_polling()


if __name__ == "__main__":
    main()