import os
import asyncio
import logging
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

# Load credentials from .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = str(os.getenv("TELEGRAM_ALLOWED_USER_ID", "")).strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHAT_MODEL = "qwen/qwen3.8-27b"
EXTRACTION_MODEL = "gemini-3.8-flash"

# In-memory sliding window for immediate turn-by-turn dialogue
chat_history = {}

# Load Persona from config/SOUL.md
SOUL_PATH = os.path.join("config", "SOUL.md")
SYSTEM_PROMPT = "You are an elite, proactive AI Personal Employee."
if os.path.exists(SOUL_PATH):
    with open(SOUL_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Mem0 Memory System
mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "collection_name": "ai_employee_memory",
            "embedding_model_dims": 384,
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": EXTRACTION_MODEL,
            "api_key": GEMINI_API_KEY,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
        },
    },
}
memory = Memory.from_config(mem0_config)

def is_authorized(update: Update) -> bool:
    user_id = str(update.effective_user.id).strip()
    return user_id == ALLOWED_USER_ID

def parse_memories(raw_output) -> list:
    """Safely normalizes both list and dict outputs from Mem0 across library versions."""
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
        "AI Personal Employee online and reporting for duty. Memory persistent on Qdrant Cloud. How can I assist you today?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return

    user_text = update.message.text.strip()
    user_id = str(update.effective_user.id).strip()

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # 1. TWO-TIER MEMORY RETRIEVAL
    all_facts = []

    # Tier 1: Pull full persistent user profile (Always injected)
    try:
        profile_res = memory.get_all(filters={"user_id": user_id})
        all_facts.extend(parse_memories(profile_res))
    except Exception as e:
        logger.warning(f"Core profile retrieval note: {e}")

    # Tier 2: Pull episodic/topic-specific memories matching this message
    try:
        search_res = memory.search(user_text, filters={"user_id": user_id})
        all_facts.extend(parse_memories(search_res))
    except Exception as e:
        logger.warning(f"Episodic memory search note: {e}")

    # Deduplicate facts while preserving order
    unique_memories = list(dict.fromkeys(all_facts))

    retrieved_context = ""
    if unique_memories:
        retrieved_context = "\n- " + "\n- ".join(unique_memories)
        print(f"\n[Active Context Loaded ({len(unique_memories)} facts)]:{retrieved_context}\n")

    # 2. CONSTRUCT SYSTEM PROMPT WITH MEMORY & ROLES
    augmented_system = SYSTEM_PROMPT
    if retrieved_context:
        augmented_system += (
            f"\n\n### KNOWN FACTS & LONG-TERM MEMORY ABOUT THE USER:\n"
            f"{retrieved_context}\n\n"
            f"Directive: Use the facts above as ground-truth knowledge about your employer. "
            f"Never claim you do not know who they are or that you have no memory if facts are listed above."
        )

    # 3. SLIDING CONVERSATION WINDOW (Last 8 turns)
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

    # Update conversation history
    chat_history[user_id].append({"role": "user", "content": user_text})
    chat_history[user_id].append({"role": "assistant", "content": reply_text})

    # 5. DELIVER RESPONSE
    await update.message.reply_text(reply_text)

    # 6. ASYNCHRONOUS BACKGROUND EXTRACTION
    def save_memory_task():
        if len(user_text.split()) < 3:
            return
        try:
            print(f"[Memory Engine] Analyzing input for extraction: '{user_text}'")
            memory.add(user_text, user_id=user_id)
            print("[Memory Engine] Saved to Qdrant Cloud successfully.")
        except Exception as e:
            logger.error(f"[Memory Engine] Extraction error: {e}")

    asyncio.get_event_loop().run_in_executor(None, save_memory_task)

def main():
    if not TELEGRAM_BOT_TOKEN or not ALLOWED_USER_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_ID in .env")

    print(f"Starting AI Employee Telegram Bot for User ID: {ALLOWED_USER_ID}...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is live! Listening for Telegram messages...")
    app.run_polling()

if __name__ == "__main__":
    main()