import os
from dotenv import load_dotenv
from groq import Groq
from mem0 import Memory

load_dotenv()

def test_groq():
    print("[1/2] Testing Groq LLM API connection...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env")

    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Respond with 'SYSTEM OPERATIONAL'"}],
        model="qwen/qwen3.8-27b",
    )
    print(f"  -> Groq Output: {chat_completion.choices[0].message.content.strip()}")

def test_cloud_memory():
    print("\n[2/2] Testing Mem0 + Qdrant Cloud connection...")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not qdrant_url or not qdrant_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing in .env")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": qdrant_url,
                "api_key": qdrant_key,
                "collection_name": "ai_employee_memory",
                "embedding_model_dims": 384,
            },
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-3.8-flash",
                "api_key": gemini_key,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "all-MiniLM-L6-v2",
            },
        },
    }

    memory = Memory.from_config(config)
    test_user_id = os.getenv("TELEGRAM_ALLOWED_USER_ID", "test_user")

    print("  -> Inserting test memory into Qdrant Cloud...")
    memory.add("Prefers daily briefings at 9:00 AM via bullet points.", user_id=test_user_id)

    print("  -> Searching Qdrant Cloud memory...")
    results = memory.search("When should briefings be sent?", filters={"user_id": test_user_id})
    print("  -> Search Result Retrieved:")
    for res in results.get("results", []):
        print(f"     * {res.get('memory')}")

if __name__ == "__main__":
    try:
        test_groq()
        test_cloud_memory()
        print("\nAll cloud connections operational!")
    except Exception as e:
        print(f"\nConnection Error: {e}")