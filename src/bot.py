import os

# Set telemetry flag BEFORE importing mem0 (mem0 reads it at import time)
os.environ["MEM0_TELEMETRY"] = "False"

import io
import re
import json
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
PORT = int(os.getenv("PORT", 10000))

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
SCHEDULE_WINDOW_MINUTES = 5   # a job still fires if the bot wakes up within this many minutes of its time

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


async def deliver_report(bot, chat_id, user_id: str, kind: str):
    records = await asyncio.to_thread(get_memory_records, user_id)
    text = await asyncio.to_thread(generate_report, kind, records)
    await send_long(bot, chat_id, text)


def due_jobs(now: datetime, last_run: dict) -> list:
    """Which scheduled reports should fire right now. Pure function so it is easy to test."""
    due = []
    today = now.date().isoformat()
    minutes_now = now.hour * 60 + now.minute

    daily_at = BRIEFING_HOUR * 60 + BRIEFING_MINUTE
    if daily_at <= minutes_now < daily_at + SCHEDULE_WINDOW_MINUTES and last_run.get("daily") != today:
        due.append("daily")

    weekly_at = WEEKLY_REPORT_HOUR * 60 + WEEKLY_REPORT_MINUTE
    if (now.weekday() == WEEKLY_REPORT_WEEKDAY
            and weekly_at <= minutes_now < weekly_at + SCHEDULE_WINDOW_MINUTES
            and last_run.get("weekly") != today):
        due.append("weekly")
    return due


async def scheduler_loop(app):
    last_run = {}
    logger.info(
        f"[Scheduler] Started. Daily briefing {BRIEFING_HOUR:02d}:{BRIEFING_MINUTE:02d}, "
        f"weekly report weekday={WEEKLY_REPORT_WEEKDAY} at {WEEKLY_REPORT_HOUR:02d}:{WEEKLY_REPORT_MINUTE:02d} "
        f"({TIMEZONE_NAME})."
    )
    while True:
        try:
            now = now_local()
            for kind in due_jobs(now, last_run):
                last_run[kind] = now.date().isoformat()
                logger.info(f"[Scheduler] Running {kind} report")
                try:
                    await deliver_report(app.bot, int(ALLOWED_USER_ID), ALLOWED_USER_ID, kind)
                except Exception as e:
                    logger.error(f"[Scheduler] {kind} report failed: {e}", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Scheduler] Loop error: {e}", exc_info=True)
        await asyncio.sleep(30)


async def post_init(app):
    if ENABLE_SCHEDULER:
        app.bot_data["scheduler_task"] = asyncio.create_task(scheduler_loop(app))


async def post_shutdown(app):
    task = app.bot_data.get("scheduler_task")
    if task:
        task.cancel()


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
        "Commands: /briefing, /weekly, /memories, /forget <number>."
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
    await process_user_text(update, context, user_text)


async def process_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    """Core assistant + memory pipeline, shared by typed messages and short voice notes."""
    user_id = str(update.effective_user.id).strip()

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

