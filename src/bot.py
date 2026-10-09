import os

# Set telemetry flag BEFORE importing mem0 (mem0 reads it at import time)
os.environ["MEM0_TELEMETRY"] = "False"

import io
import re
import json
import urllib.request
import urllib.error
import time
import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
PORT = int(os.getenv("PORT", 10000))
# Seconds to wait before polling Telegram, so the previous Render deploy has shut down first (avoids the Conflict error)
POLLING_START_DELAY = int(os.getenv("POLLING_START_DELAY", 25))

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3.8-27b")
# Small fact-extraction call on Groq (short prompt, a few hundred tokens)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", CHAT_MODEL)

# --- Audio / call recording settings ---------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip() or None  # e.g. "en" or "hi"; empty = auto-detect
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", CHAT_MODEL)
SEND_TRANSCRIPT_FILE = os.getenv("SEND_TRANSCRIPT_FILE", "true").lower() == "true"
TIMEZONE_NAME = os.getenv("TIMEZONE", "Asia/Kolkata")
# Voice notes up to this many seconds are treated as a spoken message to the assistant (answered like text).
# Longer ones are treated as call recordings (summary + action items). Set to 0 to always treat audio as recordings.
VOICE_COMMAND_MAX_SECONDS = int(os.getenv("VOICE_COMMAND_MAX_SECONDS", 60))

# --- Scheduled reports (times are in TIMEZONE, default India time) ---------
ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"
BRIEFING_HOUR = int(os.getenv("BRIEFING_HOUR", 9))
BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", 0))
WEEKLY_REPORT_WEEKDAY = int(os.getenv("WEEKLY_REPORT_WEEKDAY", 4))   # Monday=0 ... Friday=4
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR", 18))
WEEKLY_REPORT_MINUTE = int(os.getenv("WEEKLY_REPORT_MINUTE", 0))
SCHEDULE_WINDOW_MINUTES = int(os.getenv("CATCH_UP_MINUTES", 60))   # a report still goes out if the bot wakes up this many minutes after its time

# --- Web search (Tavily) ----------------------------------------------------
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", 4))
SEARCH_SNIPPET_CHARS = 700        # per result
SEARCH_TOTAL_CHARS = 3500         # all results together (keeps the prompt well under the token limit)
MAX_SEARCHES_PER_DAY = int(os.getenv("MAX_SEARCHES_PER_DAY", 30))   # protects the 1,000 free credits/month

MAX_AUDIO_BYTES = 20 * 1024 * 1024        # Telegram bots cannot download files larger than 20 MB
MAX_TRANSCRIPT_CHARS = 100_000            # safety cap on very long recordings
TRANSCRIPT_CHUNK_CHARS = int(os.getenv("TRANSCRIPT_CHUNK_CHARS", 6000))   # per analysis request
CHUNK_SPACING_SECONDS = int(os.getenv("CHUNK_SPACING_SECONDS", 15))      # pause between chunk requests
MAX_STORED_ITEMS_PER_CALL = 25

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

# Only one background memory job / audio job at a time, so jobs never pile up on the API
extraction_lock = threading.Lock()
audio_lock = threading.Lock()

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


def is_too_large_error(err: Exception) -> bool:
    text = str(err).lower()
    return "413" in text or "too large" in text


def is_temporary_error(err: Exception) -> bool:
    text = str(err)
    return any(k in text for k in ("503", "429", "UNAVAILABLE", "timeout", "timed out"))


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


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
    content = strip_thinking(resp.choices[0].message.content or "")
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
    return strip_thinking(completion.choices[0].message.content or "")


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


async def send_long(bot, chat_id, text: str):
    """Telegram messages max out at 4096 characters, so split long replies."""
    text = text or ""
    for i in range(0, len(text), 3900):
        await bot.send_message(chat_id=chat_id, text=text[i:i + 3900])


# ---------------------------------------------------------------------------
# WEB SEARCH (Tavily)
# ---------------------------------------------------------------------------
search_counter = {"day": None, "count": 0}

# Only clear "go and look this up" phrasing triggers a search automatically, so ordinary private messages
# are never sent to a third party by accident. For anything else use /search <question>.
SEARCH_TRIGGER = re.compile(
    r"\b(search\s+(for|the\s+web|online|the\s+internet)|look\s?up|google\s+(it|this|for)|"
    r"latest\s+news|news\s+(about|on)|what'?s\s+the\s+latest|find\s+out\s+(about|what|who|when))\b",
    re.IGNORECASE,
)


def wants_web_search(text: str) -> bool:
    return bool(SEARCH_TRIGGER.search(text))


def take_search_slot() -> bool:
    """Daily cap so a busy day cannot burn through the free monthly credits."""
    today = now_local().date().isoformat()
    if search_counter["day"] != today:
        search_counter["day"], search_counter["count"] = today, 0
    if search_counter["count"] >= MAX_SEARCHES_PER_DAY:
        return False
    search_counter["count"] += 1
    return True


def tavily_search(query: str):
    """Returns (results, error). Exactly one of them is meaningful."""
    if not TAVILY_API_KEY:
        return [], "the TAVILY_API_KEY is not set"

    payload = {"query": query[:380], "max_results": SEARCH_MAX_RESULTS, "search_depth": "basic"}
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("results") or [], None
    except urllib.error.HTTPError as e:
        reasons = {
            400: "the search request was rejected",
            401: "the Tavily API key is invalid",
            429: "Tavily's rate limit was hit",
            432: "the monthly Tavily credit limit was reached",
            433: "the monthly Tavily credit limit was reached",
        }
        logger.error(f"[Search] Tavily HTTP {e.code}")
        return [], reasons.get(e.code, f"Tavily returned an error (HTTP {e.code})")
    except Exception as e:
        logger.error(f"[Search] Request failed: {type(e).__name__}: {e}")
        return [], "the search service could not be reached"


def format_search_context(results: list) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()[:120]
        url = (r.get("url") or "").strip()
        content = " ".join((r.get("content") or "").split())[:SEARCH_SNIPPET_CHARS]
        blocks.append(f"[{i}] {title} ({url})\n{content}")
    body = "\n\n".join(blocks)[:SEARCH_TOTAL_CHARS]
    return (
        "\n\n### LIVE WEB SEARCH RESULTS (retrieved just now):\n" + body + "\n\n"
        "Directive: Use these results for current facts. Cite sources as [1], [2] and include a link when useful. "
        "If the results do not answer the question, say so instead of guessing. "
        "Never present anything as a web result unless it appears above."
    )


async def run_search_turn(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    """Search the web, then answer using the results AND the user's saved memory."""
    chat_id = update.effective_chat.id
    if not TAVILY_API_KEY:
        await update.message.reply_text("Web search isn't set up yet. Add TAVILY_API_KEY in Render's Environment tab.")
        return
    if not take_search_slot():
        await update.message.reply_text(
            f"I've reached today's limit of {MAX_SEARCHES_PER_DAY} web searches to protect your monthly credits. "
            "Raise MAX_SEARCHES_PER_DAY in Render if you want more."
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    results, error = await asyncio.to_thread(tavily_search, query)

    if error or not results:
        reason = error or "no results came back"
        note = f"Note: live web search didn't work ({reason}), so this answer comes from memory and general knowledge and may be out of date."
        await process_user_text(update, context, query, search_note=note, save_memory=False)
        return

    await process_user_text(
        update, context, query, search_context=format_search_context(results), save_memory=False
    )


# ---------------------------------------------------------------------------
# AUDIO / CALL RECORDING PIPELINE
# ---------------------------------------------------------------------------
def now_local() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TIMEZONE_NAME))
    except Exception:
        if TIMEZONE_NAME == "Asia/Kolkata":
            return datetime.now(timezone(timedelta(hours=5, minutes=30)))
        return datetime.now(timezone.utc)


def transcribe_audio(data: bytes, filename: str) -> str:
    """Groq Whisper transcription with retry on temporary / rate-limit errors."""
    kwargs = dict(file=(filename, data), model=WHISPER_MODEL, temperature=0)
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE

    for attempt in range(3):
        try:
            result = groq_client.audio.transcriptions.create(**kwargs)
            text = getattr(result, "text", None) or str(result)
            return text.strip()
        except Exception as e:
            if (is_rate_limit_error(e) or is_temporary_error(e)) and attempt < 2:
                wait = 20 * (attempt + 1)
                logger.warning(f"[Whisper] Limited, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def chunk_text(text: str, size: int) -> list:
    """Split a long transcript into pieces <= size chars, preferring sentence boundaries."""
    text = text.strip()
    chunks = []
    while len(text) > size:
        window = text[:size]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "),
                  window.rfind("\u0964 "), window.rfind("\n"))
        if cut < size * 0.5:      # no good boundary found: fall back to a space, then a hard cut
            cut = window.rfind(" ")
            if cut < size * 0.5:
                cut = size - 1
        chunks.append(text[:cut + 1].strip())
        text = text[cut + 1:].strip()
    if text:
        chunks.append(text)
    return chunks


def analysis_prompt(today_text: str, caption: str) -> str:
    context_line = f" The sender described it as: '{caption}'." if caption else ""
    return (
        f"You analyse a segment of a call recording or voice note transcript.{context_line} "
        f"Today's date is {today_text}. Convert relative dates (tomorrow, next Friday) into absolute dates. "
        "Reply with ONLY a JSON object with these keys: "
        '"summary" (2-3 sentences), '
        '"commitments" (list of strings: who will do what, by when), '
        '"dates" (list of strings: each date/time/deadline and what it is for), '
        '"client_requirements" (list of strings: things the client or other party asked for or needs). '
        "Use empty lists when there is nothing. Only include what is actually said; never invent details."
    )


def empty_analysis() -> dict:
    return {"summary": "", "commitments": [], "dates": [], "client_requirements": []}


def parse_analysis(content: str) -> dict:
    content = strip_thinking(content)
    result = empty_analysis()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        result["summary"] = content[:500]
        return result
    try:
        data = json.loads(match.group(0))
    except Exception:
        result["summary"] = content[:500]
        return result

    result["summary"] = str(data.get("summary", "")).strip()
    for key in ("commitments", "dates", "client_requirements"):
        value = data.get(key, [])
        if isinstance(value, str):
            value = [value]
        result[key] = [str(x).strip() for x in value if str(x).strip()]
    return result


def merge_analyses(parts: list) -> dict:
    merged = empty_analysis()
    seen = {k: set() for k in ("commitments", "dates", "client_requirements")}
    summaries = []
    for p in parts:
        if p["summary"]:
            summaries.append(p["summary"])
        for key in seen:
            for item in p[key]:
                if item.lower() not in seen[key]:
                    seen[key].add(item.lower())
                    merged[key].append(item)
    merged["summary"] = " ".join(summaries)
    return merged


def analyze_chunk(chunk: str, caption: str, today_text: str, depth: int = 0) -> dict:
    """Analyse one transcript chunk. Retries on rate limits; splits the chunk if it is too large."""
    messages = [
        {"role": "system", "content": analysis_prompt(today_text, caption)},
        {"role": "user", "content": chunk},
    ]
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=SUMMARY_MODEL, messages=messages, temperature=0, max_tokens=900
            )
            return parse_analysis(resp.choices[0].message.content or "")
        except Exception as e:
            if is_too_large_error(e) and depth < 2 and len(chunk) > 1500:
                logger.warning("[Audio] Chunk too large for the limit, splitting it in half.")
                half = chunk_text(chunk, len(chunk) // 2 + 1)
                parts = []
                for i, piece in enumerate(half):
                    if i:
                        time.sleep(CHUNK_SPACING_SECONDS)
                    parts.append(analyze_chunk(piece, caption, today_text, depth + 1))
                return merge_analyses(parts)
            if is_rate_limit_error(e) and attempt < 2:
                wait = 20 * (attempt + 1)
                logger.warning(f"[Audio] Rate limited, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise
    return empty_analysis()


def combine_summaries(summary_text: str) -> str:
    """Turn several partial summaries into one concise meeting summary."""
    try:
        resp = groq_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "Combine these partial summaries of one call into a single concise "
                                              "meeting summary of at most 6 sentences. Reply with the summary only."},
                {"role": "user", "content": summary_text[:6000]},
            ],
            temperature=0,
            max_tokens=500,
        )
        combined = strip_thinking(resp.choices[0].message.content or "")
        return combined or summary_text[:1500]
    except Exception as e:
        logger.warning(f"[Audio] Could not combine summaries: {e}")
        return summary_text[:1500]


def analyze_transcript(transcript: str, caption: str) -> dict:
    today_text = now_local().strftime("%A, %d %B %Y")
    chunks = chunk_text(transcript[:MAX_TRANSCRIPT_CHARS], TRANSCRIPT_CHUNK_CHARS)
    parts = []
    for i, chunk in enumerate(chunks):
        if i:
            time.sleep(CHUNK_SPACING_SECONDS)
        parts.append(analyze_chunk(chunk, caption, today_text))

    merged = merge_analyses(parts)
    if len(parts) > 1:
        time.sleep(CHUNK_SPACING_SECONDS)
        merged["summary"] = combine_summaries(merged["summary"])
    return merged


def store_call_memories(user_id: str, analysis: dict, caption: str) -> int:
    """Save the summary, commitments, dates and requirements as persistent memories."""
    day = now_local().strftime("%Y-%m-%d")
    label = f"Call/voice note {day}" + (f" ({caption[:40]})" if caption else "")

    facts = []
    if analysis["summary"]:
        facts.append(f"{label} - Summary: {analysis['summary'][:600]}")
    for c in analysis["commitments"]:
        facts.append(f"{label} - Commitment: {c[:300]}")
    for d in analysis["dates"]:
        facts.append(f"{label} - Date/deadline: {d[:300]}")
    for r in analysis["client_requirements"]:
        facts.append(f"{label} - Client requirement: {r[:300]}")
    facts = facts[:MAX_STORED_ITEMS_PER_CALL]

    try:
        existing = {m.strip().lower() for m in parse_memories(memory.get_all(filters={"user_id": user_id}))}
    except Exception:
        existing = set()

    stored = 0
    for fact in facts:
        if fact.strip().lower() in existing:
            continue
        for attempt in range(4):
            try:
                memory.add(fact, user_id=user_id, infer=False)
                stored += 1
                break
            except Exception as e:
                if is_temporary_error(e) and attempt < 3:
                    time.sleep(3 * (2 ** attempt))
                else:
                    logger.error(f"[Audio] Store failed: {e}")
                    break
    return stored


def format_audio_reply(analysis: dict, stored: int) -> str:
    lines = ["Summary:", analysis["summary"] or "(no summary produced)"]
    sections = (
        ("Action items / commitments", analysis["commitments"]),
        ("Dates and deadlines", analysis["dates"]),
        ("Client requirements", analysis["client_requirements"]),
    )
    for title, items in sections:
        if items:
            lines.append("")
            lines.append(f"{title}:")
            lines.extend(f"- {item}" for item in items)
    lines.append("")
    lines.append(f"Saved {stored} item(s) to long-term memory.")
    return "\n".join(lines)


async def process_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str, filename: str):
    """Short voice note: transcribe it, show the text, then answer it like a typed message."""
    bot, chat_id = context.bot, update.effective_chat.id
    try:
        tg_file = await bot.get_file(file_id)
        data = bytes(await tg_file.download_as_bytearray())
        transcript = await asyncio.to_thread(transcribe_audio, data, filename)
        if not transcript:
            await bot.send_message(chat_id=chat_id, text="I couldn't detect any speech in that voice note.")
            return
        await send_long(bot, chat_id, f"Transcribed: {transcript}")   # plain text, no Markdown parsing
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        await process_user_text(update, context, transcript)
    except Exception as e:
        logger.error(f"[Voice] Processing failed: {e}", exc_info=True)
        if is_rate_limit_error(e):
            msg = "Groq's rate limit was hit while processing that voice note. Please try again in a few minutes."
        else:
            msg = f"Sorry, I couldn't process that voice note: {e}"
        await bot.send_message(chat_id=chat_id, text=msg[:3500])


async def process_audio(bot, chat_id, user_id, file_id, filename, caption):
    """Runs in the background so the bot stays responsive during long recordings."""
    try:
        tg_file = await bot.get_file(file_id)
        data = bytes(await tg_file.download_as_bytearray())

        def pipeline():
            with audio_lock:
                transcript = transcribe_audio(data, filename)
                if not transcript:
                    return None, None, 0
                analysis = analyze_transcript(transcript, caption)
                stored = store_call_memories(user_id, analysis, caption)
                return transcript, analysis, stored

        transcript, analysis, stored = await asyncio.to_thread(pipeline)

        if transcript is None:
            await bot.send_message(chat_id=chat_id, text="I couldn't detect any speech in that recording.")
            return

        reply = format_audio_reply(analysis, stored)
        await send_long(bot, chat_id, reply)

        if SEND_TRANSCRIPT_FILE:
            buffer = io.BytesIO(transcript.encode("utf-8"))
            buffer.name = "transcript.txt"
            await bot.send_document(chat_id=chat_id, document=buffer, caption="Full transcript")

        # Keep the summary in the chat window so follow-up questions work
        history = chat_history.setdefault(user_id, [])
        history.append({"role": "user", "content": "[I sent a voice note / call recording]"})
        history.append({"role": "assistant", "content": reply[:HISTORY_ITEM_CHARS]})
        chat_history[user_id] = history[-HISTORY_MESSAGES * 2:]

    except Exception as e:
        logger.error(f"[Audio] Processing failed: {e}", exc_info=True)
        if is_rate_limit_error(e):
            msg = "Groq's rate limit was hit while processing that recording. Please try again in a few minutes."
        else:
            msg = f"Sorry, I couldn't process that recording: {e}"
        await bot.send_message(chat_id=chat_id, text=msg[:3500])


# ---------------------------------------------------------------------------
# DAILY BRIEFING + WEEKLY REPORT (own scheduler: no APScheduler / job-queue extra needed)
# ---------------------------------------------------------------------------
LABEL_COMMITMENT = " - Commitment:"
LABEL_DATE = " - Date/deadline:"
LABEL_REQUIREMENT = " - Client requirement:"
LABEL_SUMMARY = " - Summary:"
ACTION_LABELS = (LABEL_COMMITMENT, LABEL_DATE, LABEL_REQUIREMENT)
CALL_LABELS = ACTION_LABELS + (LABEL_SUMMARY,)


def parse_created_at(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_memory_records(user_id: str) -> list:
    """All stored memories for the user as dicts (id, memory, created_at), newest first."""
    try:
        raw = memory.get_all(filters={"user_id": user_id}, limit=500)
    except TypeError:
        raw = memory.get_all(filters={"user_id": user_id})

    entries = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    records = []
    for e in entries:
        if isinstance(e, dict) and e.get("memory"):
            records.append({
                "id": e.get("id"),
                "memory": e["memory"],
                "created_at": parse_created_at(e.get("created_at")),
            })
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    records.sort(key=lambda r: r["created_at"] or oldest, reverse=True)
    return records


def record_date(record):
    """The day a memory was recorded: created_at if present, else a date written in its text."""
    if record["created_at"]:
        return record["created_at"].astimezone(now_local().tzinfo).date()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", record["memory"])
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def item_line(record) -> str:
    d = record_date(record)
    has_date_in_text = re.search(r"\d{4}-\d{2}-\d{2}", record["memory"])
    prefix = f"[{d.isoformat()}] " if d and not has_date_in_text else ""
    return f"- {prefix}{record['memory'][:220]}"


def has_label(record, labels) -> bool:
    return any(label in record["memory"] for label in labels)


REPORT_INSTRUCTIONS = {
    "daily": (
        "Write a concise morning briefing for your employer. Use ONLY the saved items provided; never invent "
        "tasks, dates or people. Items show the day they were recorded. Group them under short headings "
        "(skip any heading with nothing under it): 'Due today or overdue', 'Coming up', 'Open commitments', "
        "'Client requirements to keep in mind'. Convert any relative dates using today's date. "
        "Keep it under 250 words, plain text, no markdown symbols."
    ),
    "weekly": (
        "Write a concise weekly report for your employer covering the last 7 days. Use ONLY the saved items "
        "provided; never invent anything. Use these headings (skip any with nothing under it): "
        "'Calls and conversations this week', 'Commitments made', 'Coming up next week', "
        "'Client requirements'. Keep it under 300 words, plain text, no markdown symbols."
    ),
}
REPORT_TITLES = {"daily": "Morning briefing", "weekly": "Weekly report"}
REPORT_EMPTY = {
    "daily": "Good morning. I have no open commitments or deadlines saved yet. Send me a call recording, "
             "or tell me what is on your plate, and I will track it for you.",
    "weekly": "No calls, commitments or deadlines were saved in the last 7 days, so there is nothing to report yet.",
}


def generate_report(kind: str, records: list) -> str:
    """Build the briefing/report text. Uses the LLM only when there is real data to summarise."""
    now = now_local()
    today_text = now.strftime("%A, %d %B %Y")
    title = f"{REPORT_TITLES[kind]} - {today_text}"

    if kind == "daily":
        items = [r for r in records if has_label(r, ACTION_LABELS)][:30]
        extra = [r for r in records if not has_label(r, CALL_LABELS)][:5]
    else:
        cutoff = now.date() - timedelta(days=7)
        recent = [r for r in records if has_label(r, CALL_LABELS) and (record_date(r) or date.min) >= cutoff][:40]
        recent_ids = {id(r) for r in recent}
        upcoming = [r for r in records if has_label(r, (LABEL_DATE,)) and id(r) not in recent_ids][:15]
        items = recent + upcoming
        extra = []

    if not items:
        return f"{title}\n\n{REPORT_EMPTY[kind]}"

    lines = [item_line(r) for r in items]
    background = [item_line(r) for r in extra]
    user_content = f"Today is {today_text}.\n\nSaved items:\n" + "\n".join(lines)
    if background:
        user_content += "\n\nBackground facts about your employer:\n" + "\n".join(background)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT[:2000] + "\n\n" + REPORT_INSTRUCTIONS[kind]},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(2):
        try:
            text = call_chat(messages)
            if text:
                return f"{title}\n\n{text}"
            break
        except Exception as e:
            if is_rate_limit_error(e) and attempt == 0:
                logger.warning("[Report] Rate limited, retrying in 20s")
                time.sleep(20)
            else:
                logger.error(f"[Report] AI summary failed: {e}")
                break

    # AI unavailable: still deliver the raw saved items so the report is never empty
    return f"{title}\n\n(AI summary unavailable right now. Here are your saved items.)\n" + "\n".join(lines)


async def deliver_report(bot, chat_id, user_id: str, kind: str, prefix: str = ""):
    records = await asyncio.to_thread(get_memory_records, user_id)
    text = await asyncio.to_thread(generate_report, kind, records)
    await send_long(bot, chat_id, prefix + text)


# ----- Editable schedule (change it by chatting; saved in Qdrant so it survives restarts) -----
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_LOOKUP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
SETTINGS_COLLECTION = "ai_employee_settings"
SETTINGS_POINT_ID = 1


def default_schedule() -> dict:
    return {
        "daily": {"enabled": True, "hour": BRIEFING_HOUR, "minute": BRIEFING_MINUTE},
        "weekly": {"enabled": True, "weekday": WEEKLY_REPORT_WEEKDAY,
                   "hour": WEEKLY_REPORT_HOUR, "minute": WEEKLY_REPORT_MINUTE},
    }


schedule = default_schedule()
last_run = {}          # kind -> date string of the last delivery
_qdrant_client = None


def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def load_schedule():
    """Load the saved schedule from Qdrant (falls back to the defaults if unavailable)."""
    try:
        ensure_settings_collection()
        client = get_qdrant()
        points = client.retrieve(collection_name=SETTINGS_COLLECTION, ids=[SETTINGS_POINT_ID], with_payload=True)
        if points and points[0].payload and isinstance(points[0].payload.get("schedule"), dict):
            saved = points[0].payload["schedule"]
            for kind in ("daily", "weekly"):
                if isinstance(saved.get(kind), dict):
                    for key, value in saved[kind].items():
                        if key in schedule[kind]:
                            schedule[kind][key] = value
            logger.info("[Scheduler] Loaded saved schedule from Qdrant.")
    except Exception as e:
        logger.warning(f"[Scheduler] Could not load saved schedule, using defaults: {e}")


def save_schedule():
    try:
        from qdrant_client.models import PointStruct
        get_qdrant().upsert(
            collection_name=SETTINGS_COLLECTION,
            points=[PointStruct(id=SETTINGS_POINT_ID, vector=[0.0], payload={"schedule": schedule})],
        )
        return True
    except Exception as e:
        logger.error(f"[Scheduler] Could not save schedule: {e}")
        return False


STATE_POINT_ID = 2          # holds {"last_run": {...}}
REMINDERS_POINT_ID = 3      # holds {"next_id": n, "items": [...]}
settings_collection_ready = False


def ensure_settings_collection():
    """Create the small settings collection once (it also stores the schedule and reminders)."""
    global settings_collection_ready
    if settings_collection_ready:
        return
    from qdrant_client.models import VectorParams, Distance
    client = get_qdrant()
    names = [c.name for c in client.get_collections().collections]
    if SETTINGS_COLLECTION not in names:
        client.create_collection(
            collection_name=SETTINGS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.DOT),
        )
    settings_collection_ready = True


def load_point(point_id: int) -> dict:
    try:
        ensure_settings_collection()
        points = get_qdrant().retrieve(collection_name=SETTINGS_COLLECTION, ids=[point_id], with_payload=True)
        if points and points[0].payload:
            return dict(points[0].payload)
    except Exception as e:
        logger.warning(f"[Settings] Could not load point {point_id}: {e}")
    return {}


def save_point(point_id: int, payload: dict) -> bool:
    try:
        ensure_settings_collection()
        from qdrant_client.models import PointStruct
        get_qdrant().upsert(
            collection_name=SETTINGS_COLLECTION,
            points=[PointStruct(id=point_id, vector=[0.0], payload=payload)],
        )
        return True
    except Exception as e:
        logger.error(f"[Settings] Could not save point {point_id}: {e}")
        return False


def save_state() -> bool:
    """Remember which reports were already sent today, so a restart never skips or repeats one."""
    return save_point(STATE_POINT_ID, {"last_run": dict(last_run)})


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=now_local().tzinfo)


def load_state_and_reminders():
    """Called once at startup."""
    saved = load_point(STATE_POINT_ID).get("last_run")
    if isinstance(saved, dict):
        for kind in ("daily", "weekly"):
            if isinstance(saved.get(kind), str):
                last_run[kind] = saved[kind]

    data = load_point(REMINDERS_POINT_ID)
    clean = []
    for item in data.get("items", []) if isinstance(data.get("items"), list) else []:
        try:
            if (isinstance(item, dict) and isinstance(item.get("text"), str)
                    and item.get("repeat") in (None, "daily", "weekdays", "weekly")):
                parse_iso(item["due"])
                clean.append({"id": int(item["id"]), "text": item["text"],
                              "due": item["due"], "repeat": item.get("repeat")})
        except Exception:
            continue
    reminders["items"] = clean
    reminders["next_id"] = max(int(data.get("next_id", 1) or 1), max([i["id"] for i in clean], default=0) + 1)
    logger.info(f"[Startup] Loaded {len(clean)} reminder(s); last_run={last_run}")


def reset_last_run_after_change(kind: str):
    """After the user moves a time: don't fire instantly for a time already past today."""
    now = now_local()
    entry = schedule[kind]
    scheduled = entry["hour"] * 60 + entry["minute"]
    applies_today = kind == "daily" or now.weekday() == entry.get("weekday")
    if applies_today and now.hour * 60 + now.minute >= scheduled:
        last_run[kind] = now.date().isoformat()
    else:
        last_run.pop(kind, None)


def tz_label() -> str:
    return "IST" if TIMEZONE_NAME == "Asia/Kolkata" else TIMEZONE_NAME


def fmt_time(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    return f"{(hour % 12) or 12}:{minute:02d} {suffix}"


def format_schedule() -> str:
    d, w = schedule["daily"], schedule["weekly"]
    daily = f"every day at {fmt_time(d['hour'], d['minute'])} {tz_label()}" if d["enabled"] else "turned off"
    weekly = (f"every {WEEKDAY_NAMES[w['weekday']]} at {fmt_time(w['hour'], w['minute'])} {tz_label()}"
              if w["enabled"] else "turned off")
    return (
        f"Morning briefing: {daily}\n"
        f"Weekly report: {weekly}\n\n"
        "To change them just tell me, for example: \"move my briefing to 8:30 AM\", "
        "\"send the weekly report on Monday at 5 PM\", or \"turn off the briefing\"."
    )


OFF_WORDS = ("turn off", "switch off", "disable", "stop sending", "stop the", "stop my", "pause",
             "cancel", "no more", "don't send", "dont send", "do not send")
ON_WORDS = ("turn on", "switch on", "enable", "resume", "restart", "start sending")
CHANGE_WORDS = ("change", "set ", "move", "shift", "update", "reschedule", "switch", "make it", "make the", "make my")
SEND_WORDS = ("send", "schedule", "deliver", "remind")
SHOW_WORDS = ("what time", "when do", "when is", "when will", "show schedule", "my schedule", "show my schedule")


def parse_time_of_day(t: str, kind: str):
    """Return (hour, minute, assumed) from text, None if no time, or 'invalid'."""
    if "noon" in t:
        return 12, 0, False
    if "midnight" in t:
        return 0, 0, False

    m = re.search(r"(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", t)
    if m:
        hour, minute, meridiem = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        m = re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", t)
        if m:
            hour, minute, meridiem = int(m.group(1)), 0, m.group(2)
        else:
            m = re.search(r"\b(?:at|to|by|for)\s+(\d{1,2})\b(?!\s*(?:tasks|items|days|min|things))", t)
            if not m:
                return None
            hour, minute, meridiem = int(m.group(1)), 0, None

    if minute > 59:
        return "invalid"
    assumed = False
    if meridiem:
        if not 1 <= hour <= 12:
            return "invalid"
        hour = hour % 12 + (12 if meridiem.replace(".", "").startswith("p") else 0)
    elif 1 <= hour <= 12:
        assumed = True   # no AM/PM given: morning for the briefing, evening for the weekly report
        if kind == "weekly" and hour < 12:
            hour += 12
    elif hour > 23:
        return "invalid"
    return hour, minute, assumed


def parse_schedule_request(text: str):
    """Understand 'move my briefing to 8:30 AM' style messages. Returns a dict, or None for normal chat."""
    t = " " + text.lower().strip() + " "

    kinds = set()
    if "weekly" in t or ("week" in t and "report" in t):
        kinds.add("weekly")
    if "briefing" in t or "daily report" in t or "daily summary" in t:
        kinds.add("daily")
    if not kinds and "report" in t:
        kinds.add("weekly")
    if not kinds:
        return None
    if len(kinds) == 2:
        if any(w in t for w in OFF_WORDS + ON_WORDS + CHANGE_WORDS + SEND_WORDS):
            return {"action": "ambiguous"}
        return None
    kind = kinds.pop()

    time_result = parse_time_of_day(t, kind)
    weekday_match = re.search(r"\b(" + "|".join(sorted(WEEKDAY_LOOKUP, key=len, reverse=True)) + r")\b", t)
    weekday = WEEKDAY_LOOKUP[weekday_match.group(1)] if (weekday_match and kind == "weekly") else None
    has_change_word = any(w in t for w in CHANGE_WORDS)
    has_send_word = any(w in t for w in SEND_WORDS)

    if any(w in t for w in OFF_WORDS):
        return {"action": "off", "kind": kind}

    if time_result == "invalid":
        return {"action": "invalid", "kind": kind}

    if (time_result is not None and (has_change_word or has_send_word or " at " in t or " to " in t)) or \
            (weekday is not None and has_change_word):
        change = {"action": "set", "kind": kind}
        if time_result is not None:
            change["hour"], change["minute"], change["assumed"] = time_result
        if weekday is not None:
            change["weekday"] = weekday
        return change

    if any(w in t for w in ON_WORDS):
        return {"action": "on", "kind": kind}

    if any(w in t for w in SHOW_WORDS):
        return {"action": "show", "kind": kind}
    return None


async def apply_schedule_change(change: dict) -> str:
    action = change["action"]
    if action == "ambiguous":
        return ("I can change one schedule at a time. For example: \"move my briefing to 8:30 AM\" "
                "and then \"send the weekly report on Friday at 6 PM\".")
    if action == "show":
        return format_schedule()
    if action == "invalid":
        return "I couldn't understand that time. Try something like \"8:30 AM\" or \"18:00\"."

    kind = change["kind"]
    name = "morning briefing" if kind == "daily" else "weekly report"
    entry = schedule[kind]

    if action == "off":
        entry["enabled"] = False
    elif action == "on":
        entry["enabled"] = True
    else:
        entry["enabled"] = True
        if "hour" in change:
            entry["hour"], entry["minute"] = change["hour"], change["minute"]
        if "weekday" in change:
            entry["weekday"] = change["weekday"]
    reset_last_run_after_change(kind)   # a new time that is still ahead fires today; one already past waits

    saved = await asyncio.to_thread(save_schedule)
    await asyncio.to_thread(save_state)

    if not entry["enabled"]:
        reply = f"Okay, your {name} is turned off. Say \"turn on my {name}\" any time to bring it back."
    elif kind == "daily":
        reply = f"Done. Your morning briefing will arrive every day at {fmt_time(entry['hour'], entry['minute'])} {tz_label()}."
    else:
        reply = (f"Done. Your weekly report will arrive every {WEEKDAY_NAMES[entry['weekday']]} "
                 f"at {fmt_time(entry['hour'], entry['minute'])} {tz_label()}.")
    if change.get("assumed"):
        reply += (" I assumed AM." if kind == "daily" else " I assumed PM.") + " Say the AM/PM explicitly if I got it wrong."
    if not saved:
        reply += "\n(Warning: I couldn't save this to the database, so it may reset if the bot restarts.)"
    return reply


def due_jobs(now: datetime, last_run: dict) -> list:
    """Which scheduled reports should fire right now. Pure function so it is easy to test."""
    due = []
    today = now.date().isoformat()
    minutes_now = now.hour * 60 + now.minute

    d = schedule["daily"]
    daily_at = d["hour"] * 60 + d["minute"]
    if d["enabled"] and daily_at <= minutes_now < daily_at + SCHEDULE_WINDOW_MINUTES and last_run.get("daily") != today:
        due.append("daily")

    w = schedule["weekly"]
    weekly_at = w["hour"] * 60 + w["minute"]
    if (w["enabled"] and now.weekday() == w["weekday"]
            and weekly_at <= minutes_now < weekly_at + SCHEDULE_WINDOW_MINUTES
            and last_run.get("weekly") != today):
        due.append("weekly")
    return due


def lateness_minutes(kind: str, now: datetime) -> int:
    entry = schedule[kind]
    return now.hour * 60 + now.minute - (entry["hour"] * 60 + entry["minute"])


async def scheduler_loop(app):
    logger.info(f"[Scheduler] Started.\n{format_schedule().splitlines()[0]}\n{format_schedule().splitlines()[1]}")
    attempts = {}
    while True:
        try:
            now = now_local()
            today = now.date().isoformat()
            if ENABLE_SCHEDULER:
                for kind in due_jobs(now, last_run):
                    key = f"{kind}:{today}"
                    attempts[key] = attempts.get(key, 0) + 1
                    late = lateness_minutes(kind, now)
                    prefix = f"(Sent {late} minutes late because I was offline or restarting.)\n\n" if late >= 10 else ""
                    logger.info(f"[Scheduler] Running {kind} report (attempt {attempts[key]})")
                    try:
                        await deliver_report(app.bot, int(ALLOWED_USER_ID), ALLOWED_USER_ID, kind, prefix)
                        last_run[kind] = today
                        await asyncio.to_thread(save_state)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"[Scheduler] {kind} report failed: {e}", exc_info=True)
                        if attempts[key] >= 3:       # give up for today rather than retry forever
                            last_run[kind] = today
                            await asyncio.to_thread(save_state)
            await fire_due_reminders(app.bot, now)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Scheduler] Loop error: {e}", exc_info=True)
        await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# REMINDERS: "remind me at 3 PM to call Amit". Saved in Qdrant, so they survive restarts.
# A time with AM/PM is always honoured exactly. A bare "at 3" means the next time the clock hits 3.
# ---------------------------------------------------------------------------
reminders = {"next_id": 1, "items": []}
reminders_lock = asyncio.Lock()

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
WEEKDAY_PATTERN = "|".join(sorted(WEEKDAY_LOOKUP, key=len, reverse=True))
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
PART_OF_DAY_DEFAULT = {"morning": (9, 0), "afternoon": (15, 0), "evening": (18, 0), "night": (21, 0), "tonight": (21, 0)}

REMIND_START = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok|okay|please|kindly|also|can you|could you|would you|will you)[,\s]+)*"
    r"(?:remind me|(?:set|add|create|make)\s+(?:a\s+|an\s+)?reminder)\b[\s,:\-]*",
    re.IGNORECASE,
)
LIST_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:show|list|view|see|check|display)\b.*\breminders?\b|"
    r"(?:what|which)\b.*\breminders?\b|(?:do i have|any|how many)\b.*\breminders?\b|"
    r"(?:my\s+)?reminders\s*\??\s*$)",
    re.IGNORECASE,
)
CANCEL_RE = re.compile(r"^\s*(?:please\s+)?(?:cancel|delete|remove|clear|drop)\b.*\breminders?\b", re.IGNORECASE)
REL_RE = re.compile(
    r"\bin\s+(half an?|an?|\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)\s*"
    r"(minutes?|mins?|hours?|hrs?|days?|weeks?)\b"
)


class ReminderProblem(Exception):
    """A reminder request that cannot be scheduled; the message is shown to the user."""


def fmt_due(dt: datetime, now: datetime = None) -> str:
    now = now or now_local()
    day = f"{dt.strftime('%a')} {dt.day} {dt.strftime('%b')}"
    if dt.date() == now.date():
        day = f"today, {day}"
    elif dt.date() == now.date() + timedelta(days=1):
        day = f"tomorrow, {day}"
    return f"{day} at {fmt_time(dt.hour, dt.minute)} {tz_label()}"


def repeat_label(repeat, due: datetime) -> str:
    if repeat == "daily":
        return "every day"
    if repeat == "weekdays":
        return "every weekday (Mon-Fri)"
    if repeat == "weekly":
        return f"every {WEEKDAY_NAMES[due.weekday()]}"
    return ""


def next_due(due: datetime, repeat, now: datetime) -> datetime:
    nxt = due
    while nxt <= now:
        if repeat == "daily":
            nxt += timedelta(days=1)
        elif repeat == "weekdays":
            nxt += timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
        elif repeat == "weekly":
            nxt += timedelta(days=7)
        else:
            break
    return nxt


def find_time_phrase(low: str):
    """Earliest clock time in the text: (hour, minute, meridiem or None, span) or None."""
    if (m := re.search(r"\bnoon\b", low)):
        return 12, 0, "p", m.span()
    if (m := re.search(r"\bmidnight\b", low)):
        return 12, 0, "a", m.span()
    patterns = (
        r"(?:\bat\s+|@\s*)?\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])",
        r"(?:\bat\s+)?\b(\d{1,2}):(\d{2})\b()",
        r"\bat\s+(\d{1,2})\b()()(?![:\d])(?:\s*o'?clock)?",
    )
    best = None
    for p in patterns:
        m = re.search(p, low)
        if m and (best is None or m.start() < best.start()):
            best = m
    if not best:
        return None
    hour = int(best.group(1))
    minute = int(best.group(2)) if best.group(2) else 0
    meridiem = best.group(3)[0] if best.group(3) else None
    return hour, minute, meridiem, best.span()


def resolve_calendar_date(day: int, month: int, year, today: date):
    try:
        if year:
            return date(year, month, day)
        d = date(today.year, month, day)
        return d if d >= today else date(today.year + 1, month, day)
    except ValueError:
        return None


def find_date_phrase(low: str, today: date):
    """Returns ({'kind':..., ...}, span) or (None, None). Raises ReminderProblem for impossible dates."""
    if (m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", low)):
        d = resolve_calendar_date(int(m.group(3)), int(m.group(2)), int(m.group(1)), today)
        if not d:
            raise ReminderProblem("I couldn't understand that date.")
        return {"kind": "date", "date": d}, m.span()

    if (m := re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_PATTERN})\b(?:\s*,?\s*(\d{{4}}))?", low)):
        d = resolve_calendar_date(int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)) if m.group(3) else None, today)
        if not d:
            raise ReminderProblem("I couldn't understand that date.")
        return {"kind": "date", "date": d}, m.span()

    if (m := re.search(rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{{4}}))?", low)):
        d = resolve_calendar_date(int(m.group(2)), MONTHS[m.group(1)], int(m.group(3)) if m.group(3) else None, today)
        if not d:
            raise ReminderProblem("I couldn't understand that date.")
        return {"kind": "date", "date": d}, m.span()

    if (m := re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", low)):
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        if year is not None and year < 100:
            year += 2000
        d = resolve_calendar_date(day, month, year, today) or (
            resolve_calendar_date(month, day, year, today) if month > 12 else None)
        if not d:
            raise ReminderProblem("I couldn't understand that date. I read dates as day/month, like 15/10.")
        return {"kind": "date", "date": d}, m.span()

    if (m := re.search(r"\bday after tomorrow\b", low)):
        return {"kind": "date", "date": today + timedelta(days=2)}, m.span()
    if (m := re.search(r"\btomorrow\b", low)):
        return {"kind": "date", "date": today + timedelta(days=1)}, m.span()
    if (m := re.search(r"\b(?:today|tonight)\b", low)):
        return {"kind": "today"}, m.span()

    if (m := re.search(rf"\b(?:on\s+)?(?:(next|this)\s+)?({WEEKDAY_PATTERN})\b", low)):
        return {"kind": "weekday", "weekday": WEEKDAY_LOOKUP[m.group(2)], "next": m.group(1) == "next"}, m.span()

    if (m := re.search(r"\b(?:on\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", low)):
        day = int(m.group(1))
        for offset in range(0, 4):
            month_index = today.month - 1 + offset
            year, month = today.year + month_index // 12, month_index % 12 + 1
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            if d >= today:
                return {"kind": "date", "date": d}, m.span()
        raise ReminderProblem("I couldn't understand that date.")
    return None, None


def compute_due(now: datetime, date_info, time_info, partofday, repeat):
    """Decide the exact moment. Returns (due, notes). Raises ReminderProblem or LookupError('need_time')."""
    notes = []
    today, tz = now.date(), now.tzinfo

    def at(d, hm):
        return datetime(d.year, d.month, d.day, hm[0], hm[1], tzinfo=tz)

    ambiguous = False
    if time_info:
        hour, minute, meridiem = time_info
        if minute > 59:
            raise ReminderProblem("I couldn't understand that time.")
        if meridiem:                                   # AM/PM given: honoured exactly
            if not 1 <= hour <= 12:
                raise ReminderProblem("I couldn't understand that time.")
            candidates = [(hour % 12 + (12 if meridiem == "p" else 0), minute)]
        elif hour >= 13 or hour == 0:                  # 24-hour clock
            if hour > 23:
                raise ReminderProblem("I couldn't understand that time.")
            candidates = [(hour, minute)]
        else:                                          # bare "3": ambiguous
            base = hour % 12
            if partofday in ("afternoon", "evening", "night", "tonight"):
                candidates = [(base + 12, minute)]
            elif partofday == "morning":
                candidates = [(base, minute)]
            else:
                candidates, ambiguous = [(base, minute), (base + 12, minute)], True
    elif partofday:
        candidates = [PART_OF_DAY_DEFAULT[partofday]]
        notes.append(f"You didn't give an exact time, so I used {fmt_time(*candidates[0])}.")
    elif date_info or repeat:
        candidates = [(9, 0)]
        notes.append("You didn't give a time, so I used 9:00 AM.")
    else:
        raise LookupError("need_time")

    if date_info is None:                              # time only: the next time that clock hits
        options = []
        for hm in candidates:
            dt = at(today, hm)
            if dt <= now:
                dt = at(today + timedelta(days=1), hm)
            options.append((dt, hm))
        due, chosen = min(options)
        if ambiguous:
            notes.append(f"I read it as {fmt_time(*chosen)}, the next time the clock hits that. Say AM or PM to be exact.")
        return due, notes

    kind = date_info["kind"]
    if ambiguous:
        if kind == "today":
            ahead = [hm for hm in candidates if at(today, hm) > now]
            if not ahead:
                raise ReminderProblem("That time has already passed today. Tell me a later time or add a date.")
            chosen = ahead[0]
        else:                                          # a later day: business-hours guess (7-11 AM, otherwise PM)
            base = candidates[0][0]
            chosen = candidates[0] if 7 <= base <= 11 else candidates[1]
        notes.append(f"I read it as {fmt_time(*chosen)}. Say AM or PM to be exact.")
    else:
        chosen = candidates[0]

    if kind == "today":
        d = today
    elif kind == "date":
        d = date_info["date"]
    else:
        d = today
        for _ in range(9):
            if d.weekday() == date_info["weekday"] and at(d, chosen) > now and not (date_info["next"] and d == today):
                break
            d += timedelta(days=1)

    due = at(d, chosen)
    if due <= now:
        raise ReminderProblem("That time has already passed today. Tell me a later time or add a date."
                              if d == today else "That date and time is in the past.")
    return due, notes


def extract_reminder_text(rest: str, spans: list) -> str:
    chars = list(rest)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    masked = "".join(chars)
    m = re.search(r"\b(?:to|about|that)\b\s+", masked.lower())
    text = masked[m.end():] if m else masked
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^(?:on|at|by|in|for|to|about|that)\s+", "", text, flags=re.IGNORECASE).strip(" ,.;:-")
        text = re.sub(r"\s+(?:on|at|by|in|for|to)$", "", text, flags=re.IGNORECASE).strip(" ,.;:-")
    return text[:1].upper() + text[1:] if text else ""


def parse_new_reminder(rest: str, now: datetime) -> dict:
    low = rest.lower()
    masked = list(low)
    spans = []

    def mask(span):
        spans.append(span)
        for i in range(span[0], span[1]):
            masked[i] = " "

    def cur():
        return "".join(masked)

    try:
        # repeat
        repeat, repeat_weekday = None, None
        if (m := re.search(r"\bevery\s+weekdays?\b|\bon\s+weekdays\b|\bweekdays\s+(?=at\b|\d)", cur())):
            repeat = "weekdays"; mask(m.span())
        elif (m := re.search(rf"\bevery\s+({WEEKDAY_PATTERN})s?\b", cur())):
            repeat, repeat_weekday = "weekly", WEEKDAY_LOOKUP[m.group(1)]; mask(m.span())
        elif (m := re.search(r"\bevery\s*day\b|\beach\s+day\b|^\s*daily\b|\bdaily\s+(?=at\b|\d)", cur())):
            repeat = "daily"; mask(m.span())
        elif (m := re.search(r"\bevery\s+week\b|^\s*weekly\b|\bweekly\s+(?=at\b|\d)", cur())):
            repeat = "weekly"; mask(m.span())

        # relative ("in 30 minutes", "in 2 days")
        rel_due, date_info = None, None
        if (m := REL_RE.search(cur())):
            word, unit = m.group(1), m.group(2)
            qty = 0.5 if word.startswith("half") else (NUM_WORDS.get(word) or float(word))
            if qty <= 0:
                raise ReminderProblem("Tell me a time in the future, for example \"in 30 minutes\".")
            mask(m.span())
            if unit.startswith("m"):
                rel_due = now + timedelta(minutes=qty)
            elif unit.startswith("h"):
                rel_due = now + timedelta(hours=qty)
            else:
                days = int(qty * (7 if unit.startswith("w") else 1))
                date_info = {"kind": "date", "date": now.date() + timedelta(days=days)}

        if rel_due is not None:
            due = rel_due.replace(second=0, microsecond=0)
            notes = []
            time_info = partofday = None
        else:
            # "this morning/afternoon/evening" and "tonight" mean today
            partofday = None
            if (m := re.search(r"\bthis\s+(morning|afternoon|evening)\b", cur())):
                partofday, date_info = m.group(1), {"kind": "today"}; mask(m.span())
            elif (m := re.search(r"\btonight\b", cur())):
                partofday, date_info = "tonight", {"kind": "today"}; mask(m.span())

            if date_info is None:
                date_info, span = find_date_phrase(cur(), now.date())
                if span:
                    mask(span)
                    pm = re.compile(r"\s*(?:in\s+the\s+)?(morning|afternoon|evening|night)\b").match(cur(), span[1])
                    if pm:
                        partofday = pm.group(1); mask((span[1], pm.end()))
            if partofday is None and (m := re.search(r"\bin\s+the\s+(morning|afternoon|evening)\b", cur())):
                partofday = m.group(1); mask(m.span())

            if repeat == "weekly" and repeat_weekday is not None and date_info is None:
                date_info = {"kind": "weekday", "weekday": repeat_weekday, "next": False}

            time_found = find_time_phrase(cur())
            time_info = None
            if time_found:
                time_info = time_found[:3]
                mask(time_found[3])

            try:
                due, notes = compute_due(now, date_info, time_info, partofday, repeat)
            except LookupError:
                return {"action": "need_time"}
    except ReminderProblem as e:
        return {"action": "error", "message": str(e)}

    if repeat == "weekdays":
        while due.weekday() >= 5:
            due += timedelta(days=1)

    text = extract_reminder_text(rest, spans)
    if not text:
        return {"action": "need_text"}
    return {"action": "add", "text": text[:300], "due": due, "repeat": repeat, "notes": notes}


def parse_reminder_request(text: str, now: datetime = None):
    """Returns a dict for reminder requests (add / list / cancel), or None for ordinary messages."""
    now = now or now_local()
    t = text.strip()
    m = REMIND_START.match(t)
    if m:
        return parse_new_reminder(t[m.end():], now)
    if CANCEL_RE.match(t):
        low = t.lower()
        if re.search(r"\ball\b", low):
            return {"action": "cancel_all"}
        idm = re.search(r"(?:#|\bnumber\s+|\bno\.?\s*|\breminders?\s+)(\d+)", low) or re.search(r"\b(\d+)\b", low)
        return {"action": "cancel", "id": int(idm.group(1))} if idm else {"action": "cancel_none"}
    if LIST_RE.match(t):
        return {"action": "list"}
    return None


def reminder_snapshot() -> dict:
    return {"next_id": reminders["next_id"], "items": [dict(i) for i in reminders["items"]]}


def save_reminders(snapshot: dict) -> bool:
    return save_point(REMINDERS_POINT_ID, snapshot)


def format_reminders() -> str:
    items = sorted(reminders["items"], key=lambda r: parse_iso(r["due"]))
    if not items:
        return "You have no reminders. Try: \"remind me tomorrow at 10 AM to send the quote\"."
    now = now_local()
    lines = []
    for r in items:
        due = parse_iso(r["due"])
        extra = f" (repeats {repeat_label(r.get('repeat'), due)})" if r.get("repeat") else ""
        lines.append(f"#{r['id']} - {fmt_due(due, now)} - {r['text']}{extra}")
    return "Your reminders:\n" + "\n".join(lines) + "\n\nCancel one with \"cancel reminder 3\" or all with \"cancel all reminders\"."


async def apply_reminder_request(change: dict) -> str:
    action = change["action"]
    if action == "need_time":
        return ("When should I remind you? For example: \"remind me at 3 PM to call Amit\" or "
                "\"remind me tomorrow at 10 AM to send the quote\".")
    if action == "need_text":
        return "What should I remind you about? For example: \"remind me at 3 PM to call Amit\"."
    if action == "error":
        return change["message"]
    if action == "list":
        return format_reminders()
    if action == "cancel_none":
        return "Which reminder? Say \"cancel reminder 3\" using a number from your list.\n\n" + format_reminders()

    async with reminders_lock:
        if action == "cancel_all":
            count = len(reminders["items"])
            reminders["items"].clear()
            saved = await asyncio.to_thread(save_reminders, reminder_snapshot())
            reply = f"Cancelled {count} reminder(s)." if count else "You have no reminders to cancel."
        elif action == "cancel":
            match = next((r for r in reminders["items"] if r["id"] == change["id"]), None)
            if not match:
                return f"I couldn't find reminder #{change['id']}.\n\n" + format_reminders()
            reminders["items"].remove(match)
            saved = await asyncio.to_thread(save_reminders, reminder_snapshot())
            reply = f"Cancelled reminder #{match['id']}: {match['text']}"
        else:  # add
            item = {"id": reminders["next_id"], "text": change["text"],
                    "due": change["due"].isoformat(), "repeat": change["repeat"]}
            reminders["next_id"] += 1
            reminders["items"].append(item)
            saved = await asyncio.to_thread(save_reminders, reminder_snapshot())
            reply = f"Reminder #{item['id']} set for {fmt_due(change['due'])}: {item['text']}."
            if change["repeat"]:
                reply += f" Repeats {repeat_label(change['repeat'], change['due'])}."
            for note in change["notes"]:
                reply += f" ({note})"

    if not saved:
        reply += "\n(Warning: I couldn't save this to the database, so it may be lost if the bot restarts.)"
    return reply


async def fire_due_reminders(bot, now: datetime):
    """Send every reminder that is due. Reminders missed during downtime are sent late, never lost."""
    due_items = [r for r in list(reminders["items"]) if parse_iso(r["due"]) <= now]
    for item in due_items:
        try:
            due_dt = parse_iso(item["due"])
            text = f"Reminder: {item['text']}"
            if (now - due_dt) >= timedelta(minutes=10):
                text += f"\n(This was due {fmt_due(due_dt, now)}. I was offline or restarting.)"
            await send_long(bot, int(ALLOWED_USER_ID), text)
            async with reminders_lock:
                if item in reminders["items"]:
                    if item.get("repeat"):
                        item["due"] = next_due(due_dt, item["repeat"], now).isoformat()
                    else:
                        reminders["items"].remove(item)
                    await asyncio.to_thread(save_reminders, reminder_snapshot())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Reminders] Could not deliver reminder #{item.get('id')}: {e}", exc_info=True)


async def post_init(app):
    # The health server is already up, so Render marks this deploy healthy and stops the old copy meanwhile.
    # Polling and the scheduler start only after that, so two copies never poll or send reminders together.
    if POLLING_START_DELAY > 0:
        logger.info(f"[Startup] Waiting {POLLING_START_DELAY}s for the previous instance to shut down.")
        await asyncio.sleep(POLLING_START_DELAY)
    await asyncio.to_thread(load_schedule)
    await asyncio.to_thread(load_state_and_reminders)
    app.bot_data["scheduler_task"] = asyncio.create_task(scheduler_loop(app))


async def post_shutdown(app):
    task = app.bot_data.get("scheduler_task")
    if task:
        task.cancel()
        try:
            await task   # let the cancellation finish so no "Task was destroyed but it is pending" noise
        except BaseException:
            pass


async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await update.message.reply_text(format_reminders())


async def cancel_reminder_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    arg = (context.args[0] if context.args else "").lower().lstrip("#")
    if arg == "all":
        change = {"action": "cancel_all"}
    elif arg.isdigit():
        change = {"action": "cancel", "id": int(arg)}
    else:
        change = {"action": "cancel_none"}
    await update.message.reply_text(await apply_reminder_request(change))


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await update.message.reply_text(format_schedule())


async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    user_id = str(update.effective_user.id).strip()
    await deliver_report(context.bot, update.effective_chat.id, user_id, "daily")


async def weekly_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    user_id = str(update.effective_user.id).strip()
    await deliver_report(context.bot, update.effective_chat.id, user_id, "weekly")


async def memories_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List the newest saved memories so you can remove outdated ones with /forget <number>."""
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    user_id = str(update.effective_user.id).strip()
    try:
        records = await asyncio.to_thread(get_memory_records, user_id)
    except Exception as e:
        await update.message.reply_text(f"Could not read memory: {e}")
        return
    if not records:
        await update.message.reply_text("No memories saved yet.")
        return

    shown = records[:20]
    context.application.bot_data["memory_listing"] = [r["id"] for r in shown]
    lines = [f"{i}. {r['memory'][:150]}" for i, r in enumerate(shown, start=1)]
    header = f"Newest {len(shown)} of {len(records)} memories:\n\n"
    footer = "\n\nRemove one with /forget <number>, for example /forget 3."
    await send_long(context.bot, update.effective_chat.id, header + "\n".join(lines) + footer)


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    listing = context.application.bot_data.get("memory_listing")
    if not listing:
        await update.message.reply_text("Run /memories first, then use /forget <number>.")
        return
    try:
        index = int(context.args[0])
        memory_id = listing[index - 1]
        if index < 1 or not memory_id:
            raise ValueError
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /forget <number> using a number from /memories.")
        return
    try:
        await asyncio.to_thread(memory.delete, memory_id)
        listing[index - 1] = None   # that number can no longer be reused by mistake
        await update.message.reply_text(f"Deleted memory {index}.")
    except Exception as e:
        await update.message.reply_text(f"Could not delete it: {e}")


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    await update.message.reply_text(
        "AI Personal Employee operational on Render Cloud. How can I assist you today?\n"
        "Send a short voice note and I'll answer it like a message. Send a longer voice note or call recording "
        "and I'll transcribe it, summarise it, and save the action items.\n"
        "Commands: /search <question>, /reminders, /briefing, /weekly, /schedule, /memories, /forget <number>.\n"
        "Reminders: say \"remind me at 3 PM to call Amit\" or \"remind me tomorrow at 10 AM to send the quote\".\n"
        "You can also say things like \"move my briefing to 8:30 AM\"."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return

    msg = update.message
    if msg.voice:
        media, filename = msg.voice, "voice.ogg"
    elif msg.audio:
        media, filename = msg.audio, (msg.audio.file_name or "audio.mp3")
    else:
        media, filename = msg.document, (msg.document.file_name or "recording.m4a")

    if media.file_size and media.file_size > MAX_AUDIO_BYTES:
        await msg.reply_text(
            "That file is over 20 MB, which is the most a Telegram bot can download. "
            "Please trim or compress the recording (for example split it into parts) and send it again."
        )
        return

    user_id = str(update.effective_user.id).strip()
    caption = (msg.caption or "").strip()

    is_short_voice = (
        bool(msg.voice)
        and VOICE_COMMAND_MAX_SECONDS > 0
        and (msg.voice.duration or 0) <= VOICE_COMMAND_MAX_SECONDS
        and not caption
    )

    if is_short_voice:
        # Short voice note -> treat as a spoken message to the assistant
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        context.application.create_task(
            process_voice_command(update, context, media.file_id, filename)
        )
        return

    # Long voice note / audio file / call recording -> transcript, summary, action items, memory
    await msg.reply_text("Got it. Transcribing now. Long recordings can take a few minutes.")
    context.application.create_task(
        process_audio(context.bot, update.effective_chat.id, user_id, media.file_id, filename, caption)
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return

    user_text = update.message.text.strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Schedule changes are handled first; otherwise explicit "search for ..." style requests use the web
    if (parse_reminder_request(user_text) is None and parse_schedule_request(user_text) is None
            and wants_web_search(user_text)):
        await run_search_turn(update, context, user_text[:300])
        return

    await process_user_text(update, context, user_text)


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("Access restricted.")
        return
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /search <what to look up>, for example /search latest GST rate changes")
        return
    await run_search_turn(update, context, query[:300])


async def process_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str,
                            search_context: str = "", search_note: str = "", save_memory: bool = True):
    """Core assistant + memory pipeline, shared by typed messages and short voice notes."""
    user_id = str(update.effective_user.id).strip()

    # Reminders ("remind me at 3 PM to call Amit") are handled here, not sent to the AI or saved as a memory
    reminder_change = parse_reminder_request(user_text)
    if reminder_change is not None:
        await update.message.reply_text(await apply_reminder_request(reminder_change))
        return

    # "Move my briefing to 8:30 AM" etc. is handled here, not sent to the AI or saved as a memory
    change = parse_schedule_request(user_text)
    if change is not None:
        await update.message.reply_text(await apply_schedule_change(change))
        return

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

    if search_context:
        augmented_system += search_context

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
    if search_note:
        reply_text = f"{search_note}\n\n{reply_text}"
    await update.message.reply_text(reply_text)

    # 6. BACKGROUND MEMORY SAVE (skipped for web-search turns so lookups don't clutter memory)
    if not save_memory:
        return
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
    if type(context.error).__name__ == "Conflict":
        logger.warning("[Telegram] Another copy of the bot is polling (normal for a few seconds during a deploy). "
                       "If this repeats every few seconds, a second instance is running somewhere.")
        return
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


def build_audio_filter():
    """Voice notes, audio files, and call recordings sent as documents (.m4a, .mp3, ...)."""
    audio_filter = filters.VOICE | filters.AUDIO | filters.Document.AUDIO
    for ext in ("m4a", "mp3", "wav", "ogg", "flac", "webm"):
        audio_filter = audio_filter | filters.Document.FileExtension(ext)
    return audio_filter


def main():
    if not TELEGRAM_BOT_TOKEN or not ALLOWED_USER_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_ID")

    http_thread = threading.Thread(target=run_health_server, daemon=True)
    http_thread.start()
    print(f"Health server successfully bound to port {PORT}")

    print(f"Starting AI Employee for User ID: {ALLOWED_USER_ID}...")
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("reminders", reminders_command))
    app.add_handler(CommandHandler("cancelreminder", cancel_reminder_command))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("briefing", briefing_command))
    app.add_handler(CommandHandler("weekly", weekly_command))
    app.add_handler(CommandHandler("memories", memories_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(MessageHandler(build_audio_filter(), handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(error_handler)

    print("Bot is live! Listening for Telegram messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()