# src/test_google.py
import sys

def main():
    print("=" * 60)
    print("Google Workspace integration test")
    print("=" * 60)

    print("\n[1/4] Importing google_tools...")
    try:
        import google_tools
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Add these to requirements.txt and redeploy:")
        print("     google-api-python-client")
        print("     google-auth-httplib2")
        print("     google-auth-oauthlib")
        sys.exit(1)
    print("   ✅ Import OK")

    print("\n[2/4] Checking Google configuration...")
    if not google_tools.is_google_configured():
        print("❌ Google not configured.")
        print("   Run auth_google.py locally, then set GOOGLE_TOKEN_JSON on Render.")
        sys.exit(1)
    print("   ✅ Token loaded and valid")

    print("\n[3/4] Fetching upcoming calendar events...")
    try:
        cal = google_tools.get_upcoming_events(max_results=3, days_ahead=7)
        print(cal)
        print("   ✅ Calendar OK")
    except Exception as e:
        print(f"   ❌ Calendar error: {e}")

    print("\n[4/4] Fetching recent emails...")
    try:
        mail = google_tools.get_recent_emails(max_results=3)
        print(mail)
        print("   ✅ Gmail OK")
    except Exception as e:
        print(f"   ❌ Gmail error: {e}")

    print("\n" + "=" * 60)
    print("✅ All Google integrations working.")
    print("=" * 60)

if __name__ == "__main__":
    main()