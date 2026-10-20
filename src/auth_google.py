# auth_google.py
import os
import sys

def main():
    # Check if the secret file exists
    if not os.path.exists("client_secret.json"):
        print("❌ client_secret.json not found in this folder.")
        print("   Make sure you downloaded it from Google Cloud Console and placed it here.")
        sys.exit(1)

    # The exact scopes the bot needs
    scopes = [
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/gmail.readonly",
    ]

    print("🔑 Starting authorization...")
    print("   Your default web browser will open automatically.\n")
    
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", scopes)
        
        # port=0 lets the OS pick an available port automatically.
        # This prevents "port already in use" errors and handles the Desktop app redirect correctly.
        creds = flow.run_local_server(port=0)

        # Save the token
        with open("token.json", "w") as token:
            token.write(creds.to_json())

        print("\n✅ Success! token.json has been created.")
        print("\nNext steps:")
        print("1. Open token.json in a text editor (like Notepad).")
        print("2. Copy its ENTIRE contents (it's one long JSON object).")
        print("3. Go to your Render dashboard -> Environment -> Add Environment Variable.")
        print("4. Name it exactly: GOOGLE_TOKEN_JSON")
        print("5. Paste the contents and save.")

    except Exception as e:
        print(f"\n❌ Authorization failed: {e}")
        print("Double-check that your client_secret.json was created as a 'Desktop app' (not Web application).")
        sys.exit(1)

if __name__ == "__main__":
    main()