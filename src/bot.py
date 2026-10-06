import os

# Set telemetry flag BEFORE importing mem0 (mem0 reads it at import time)
os.environ["MEM0_TELEMETRY"] = "False"

import re
import json
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

# ---------------------------------------------------------------------------
# Logging (httpx/httpcore at WARNING so the Telegram token is never printed)
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credentials and settings
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = str(os.getenv("TELEGRAM_ALLOWED_USER_ID", "")).strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PORT = int(os.getenv("PORT", 10000))

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3.8-27b")
# Small fact-extraction call on Groq (short prompt, a few hundred tokens)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", CHAT_MODEL)

EMBEDDING_DIMS = 768

# ---------------------------------------------------------------------------
# TOKEN BUDGET (keeps every Groq request far below the per-minute limit)
# ---------------------------------------------------------------------------
MAX_USER_CHARS = 4000        # longest user message sent to the model
MAX_SYSTEM_CHARS = 6000      # cap on SOUL.md / persona size (~1,500 tokens)
MAX_MEMORY_ITEMS = 15        # max stored facts injected into the prompt
MAX_MEMORY_ITEM_CHARS = 250  # max length of each injected fact
HISTORY_MESSAGES = 6         # last N chat messages kept in the prompt
HISTORY_ITEM_CHARS = 1200    # max length of each history message
CHAT_MAX_OUTPUT_TOKENS = 1024

# In-memory sliding window for turn-by-turn dialogue
chat_history = {}

# Only one background memory job at a time, so jobs never pile up on the API
extraction_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------
SOUL_PATH = os.path.join("config", "SOUL.md")
SYSTEM_PROMPT = "You are an elite, proactive AI Personal Employee."
if os.path.exists(SOUL_PATH):
    with open(SOUL_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
SYSTEM_PROMPT = SYSTEM_PROMPT[:MAX_SYSTEM_CHARS]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY)

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
    # NOTE: mem0's own LLM is NOT called anymore (we store with infer=False),
    # so its large ~8,000-token extraction prompt is never sent.
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-3.8-flash",
            "api_key": GEMINI_API_KEY,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "embedding_dims": EMBEDDING_DIMS,
            "api_key": GEMINI_API_KEY,
        },
    },
}

memory = Memory.from_config(mem0_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_authorized(update: Update) -> bool:
    if not update or not update.effective_user:
        return False
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


def is_rate_limit_error(err: Exception) -> bool:
    text = str(err).lower()
    return any(k in text for k in ("413", "429", "rate_limit", "rate limit", "too large"))


def is_temporary_error(err: Exception) -> bool:
    text = str(err)
    return any(k in text for k in ("503", "429", "UNAVAILABLE", "timeout", "timed out"))


EXTRACTION_PROMPT = (
    "Extract lasting facts worth remembering about the user from their message: "
    "personal details, preferences, goals, people, projects, schedules, decisions. "
    "Ignore greetings, small talk, and one-off questions. "
    "Reply with ONLY a JSON array of short, self-contained strings, e.g. "
    '["User runs a clothing business in Jaipur"]. If nothing is worth remembering, reply [].'
)


def extract_facts(text: str) -> list:
    """Tiny extraction call (a few hundred tokens) instead of mem0's ~8,000-token prompt."""
    resp = groq_client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text[:MAX_USER_CHARS]},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = resp.choices[0].message.content or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    match = re.search(r"\[.*\]", content, flags=re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group(0))
    facts = [str(x).strip() for x in data if isinstance(x, str) and x.strip()]
    return facts[:5]


def build_memory_context(memories: list) -> str:
    """Cap how many facts (and how long) go into the prompt."""
    trimmed = [m[:MAX_MEMORY_ITEM_CHARS] for m in memories[:MAX_MEMORY_ITEMS]]
    return "\n- " + "\n- ".join(trimmed) if trimmed else ""


def call_chat(messages: list) -> str:
    completion = groq_client.chat.completions.create(
        messages=messages,
        model=CHAT_MODEL,
        max_tokens=CHAT_MAX_OUTPUT_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()


def chat_with_fallback(messages_payload: list, user_text: str) -> str:
    """Normal call; on a rate-limit error wait and retry with a minimal prompt."""
    try:
        return call_chat(messages_payload)
    except Exception as e:
        if not is_rate_limit_error(e):
            return f"Processing error: {e}"
        logger.warning(f"[Chat] Rate limit hit, retrying with a minimal prompt: {e}")

    time.sleep(8)
    minimal = [
        {"role": "system", "content": SYSTEM_PROMPT[:1500]},
        {"role": "user", "content": user_text[:MAX_USER_CHARS]},
    ]
    try:
        return call_chat(minimal)
    except Exception as e:
        logger.error(f"[Chat] Still rate limited: {e}")
        return "I'm briefly rate limited by my AI provider. Please send that again in about a minute."


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
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

    # 1. TWO-TIER RETRIEVAL (run off the event loop so the bot stays responsive)
    all_facts = []

    try:
        profile_res = await asyncio.to_thread(memory.get_all, filters={"user_id": user_id})
        all_facts.extend(parse_memories(profile_res))
    except Exception as e:
        logger.warning(f"Profile retrieval: {e}")

    try:
        search_res = await asyncio.to_thread(memory.search, user_text, filters={"user_id": user_id})
        all_facts.extend(parse_memories(search_res))
    except Exception as e:
        logger.warning(f"Episodic search: {e}")

    unique_memories = list(dict.fromkeys(all_facts))
    retrieved_context = build_memory_context(unique_memories)

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
    messages_payload.extend(chat_history[user_id][-HISTORY_MESSAGES:])
    messages_payload.append({"role": "user", "content": user_text[:MAX_USER_CHARS]})

    # 4. INFERENCE VIA GROQ (with rate-limit fallback)
    reply_text = await asyncio.to_thread(chat_with_fallback, messages_payload, user_text)

    chat_history[user_id].append({"role": "user", "content": user_text[:HISTORY_ITEM_CHARS]})
    chat_history[user_id].append({"role": "assistant", "content": reply_text[:HISTORY_ITEM_CHARS]})
    chat_history[user_id] = chat_history[user_id][-HISTORY_MESSAGES * 2:]

    # 5. SEND REPLY
    await update.message.reply_text(reply_text)

    # 6. BACKGROUND MEMORY SAVE
    known = {m.strip().lower() for m in unique_memories}

    def save_memory_task():
        if len(user_text.split()) < 3:
            return

        with extraction_lock:
            # --- Step A: extract facts with the tiny prompt (retry on temporary limits)
            facts = None
            for attempt in range(3):
                try:
                    facts = extract_facts(user_text)
                    break
                except Exception as e:
                    if (is_rate_limit_error(e) or is_temporary_error(e)) and attempt < 2:
                        wait = 10 * (attempt + 1)
                        logger.warning(f"[Memory Engine] Extraction limited, retrying in {wait}s")
                        time.sleep(wait)
                    else:
                        logger.error(f"[Memory Engine] Extraction failed: {e}")
                        break

            if facts is None:
                # Extraction unavailable: keep the raw message so nothing is lost
                facts = [user_text[:500]] if len(user_text.split()) >= 5 else []

            facts = [f for f in facts if f.strip().lower() not in known]
            if not facts:
                logger.info("[Memory Engine] Nothing new to store.")
                return

            # --- Step B: store directly (infer=False -> no mem0 LLM call, no big prompt)
            for fact in facts:
                for attempt in range(4):
                    try:
                        memory.add(fact, user_id=user_id, infer=False)
                        logger.info(f"[Memory Engine] Stored: {fact[:80]}")
                        break
                    except Exception as e:
                        if is_temporary_error(e) and attempt < 3:
                            time.sleep(3 * (2 ** attempt))
                        else:
                            logger.error(f"[Memory Engine] Store failed: {e}")
                            break

    asyncio.get_running_loop().run_in_executor(None, save_memory_task)


# Global Telegram Error Handler to safely catch and log unhandled exceptions
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


# ---------------------------------------------------------------------------
# Health check server (keeps Render's port binding happy)
# ---------------------------------------------------------------------------
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

    http_thread = threading.Thread(target=run_health_server, daemon=True)
    http_thread.start()
    print(f"Health server successfully bound to port {PORT}")

    print(f"Starting AI Employee for User ID: {ALLOWED_USER_ID}...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(error_handler)

    print("Bot is live! Listening for Telegram messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

