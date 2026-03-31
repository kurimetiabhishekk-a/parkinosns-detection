

import os
import sys

def main():
    from pyngrok import ngrok, conf

    token = os.environ.get("NGROK_AUTHTOKEN", "")
    if not token:
        print("\n" + "="*55)
        print("  ParkiSense – Public URL Setup")
        print("="*55)
        print("\n  To get your FREE authtoken:")
        print("  1. Go to https://ngrok.com and create a free account")
        print("  2. Go to https://dashboard.ngrok.com/get-started/your-authtoken")
        print("  3. Copy the token and paste it below\n")
        token = input("  Paste your ngrok authtoken here: ").strip()
        if not token:
            print("\n[ERROR] No token provided. Exiting.")
            sys.exit(1)

    conf.get_default().auth_token = token
    ngrok.kill()  # kill any existing tunnels

    print("\n[1/2] Starting Flask app on port 5000...")

    import threading
    import main as flask_app

    def run_flask():
        flask_app.app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

    t = threading.Thread(target=run_flask, daemon=True)
    t.start()

    import time
    time.sleep(3)  # Wait for Flask to start

    print("[2/2] Opening public tunnel...")
    tunnel = ngrok.connect(5000, "http")
    public_url = tunnel.public_url

    print("\n" + "="*55)
    print("  ✅ ParkiSense is LIVE!")
    print(f"  🌐 Public URL: {public_url}")
    print("="*55)
    print("\n  Share this URL with anyone — it works in any browser!")
    print("  ⚠️  URL is active as long as this window is open.")
    print("  Press Ctrl+C to stop.\n")

    try:
        ngrok.get_ngrok_process().proc.wait()
    except KeyboardInterrupt:
        print("\n[Stopped] Closing tunnel.")
        ngrok.kill()

if __name__ == "__main__":
    main()
