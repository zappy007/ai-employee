#!/usr/bin/env python3
"""One-time helper that gets a Google refresh token for the AI Employee bot.

Run it on YOUR OWN PC (not on the server). It uses only the Python standard library.
It opens a browser, you sign in and approve, and it prints a refresh token to paste into Render.

Permissions requested:
  - Gmail: read-only (the bot can never send, delete or change email)
  - Calendar: view and create events (the bot only creates events after you reply "yes")
"""
import base64
import getpass
import hashlib
import http.server
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


def make_pkce():
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(client_id: str, redirect_uri: str, challenge: str, state: str) -> str:
    return AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",     # needed to receive a refresh token
        "prompt": "consent",          # forces Google to issue a new refresh token
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    })


def make_server(state: str, result: dict):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in query and query.get("state", [None])[0] == state:
                result["code"] = query["code"][0]
                body, status = "Done. You can close this tab and go back to the terminal.", 200
            elif "error" in query or "code" in query:
                result["error"] = query.get("error", ["the response did not match this sign-in"])[0]
                body, status = "Sign-in did not complete. Go back to the terminal.", 400
            else:
                body, status = "Waiting for Google...", 404     # e.g. a favicon request
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    return http.server.HTTPServer(("127.0.0.1", 0), Handler)


def wait_for_code(server, result: dict):
    while "code" not in result and "error" not in result:
        server.handle_request()


def exchange_code(client_id, client_secret, code, redirect_uri, verifier) -> dict:
    data = urllib.parse.urlencode({
        "client_id": client_id, "client_secret": client_secret, "code": code,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data, method="POST"), timeout=20) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Google rejected the sign-in (HTTP {e.code}): {e.read().decode('utf-8', 'ignore')[:300]}")


def main():
    print(__doc__)
    client_id = os.getenv("GOOGLE_CLIENT_ID") or input("Client ID: ").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or getpass.getpass("Client secret (typing is hidden): ").strip()
    if not client_id or not client_secret:
        raise SystemExit("Both the client ID and the client secret are required.")

    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(16)
    result = {}
    server = make_server(state, result)
    redirect_uri = f"http://127.0.0.1:{server.server_port}"
    url = build_auth_url(client_id, redirect_uri, challenge, state)

    print("\nOpening your browser. If nothing opens, copy this link into it:\n")
    print(url + "\n")
    print('Google will warn that the app is "unverified". That is expected for your own app:')
    print('click "Advanced", then "Go to <your app name> (unsafe)", and approve both permissions.\n')
    webbrowser.open(url)
    wait_for_code(server, result)
    server.server_close()
    if "error" in result:
        raise SystemExit(f"Sign-in failed: {result['error']}")

    tokens = exchange_code(client_id, client_secret, result["code"], redirect_uri, verifier)
    granted = tokens.get("scope", "")
    missing = [s.rsplit("/", 1)[-1] for s in SCOPES if s not in granted]
    refresh = tokens.get("refresh_token")
    if not refresh:
        raise SystemExit("Google did not return a refresh token. Remove this app at "
                         "https://myaccount.google.com/permissions and run the script again.")
    if missing:
        print(f"WARNING: you did not grant: {', '.join(missing)}. Run again and tick every permission box.\n")

    print("Success. Add these three values in Render (Environment tab), then redeploy:\n")
    print(f"GOOGLE_CLIENT_ID={client_id}")
    print("GOOGLE_CLIENT_SECRET=<the client secret you entered>")
    print(f"GOOGLE_REFRESH_TOKEN={refresh}\n")
    print("Treat the refresh token like a password: do not paste it in chats, screenshots or GitHub.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")