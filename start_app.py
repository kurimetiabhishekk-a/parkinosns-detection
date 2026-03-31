import os
import sys
import subprocess
import time
from dotenv import load_dotenv

def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{icons.get(level, '🔹')} {msg}")

def check_dependencies():
    log("Checking dependencies...")
    try:
        import flask, pymongo, cryptography, dotenv
        log("Core libraries found.", "SUCCESS")
    except ImportError as e:
        log(f"Missing library: {e}. Installing now...", "WARNING")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_app():

    load_dotenv()

    if not os.path.exists(".env"):
        log("No .env file found! Creating one with defaults...", "WARNING")

        return

    uri = os.environ.get('MONGODB_URI', '').strip()
    if not uri or '<password>' in uri:
        log("MONGODB_URI is not set correctly in your .env file.", "ERROR")
        return

    log("Testing Database connection...")
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command('ping')
        log("Database connection ACTIVE!", "SUCCESS")
    except Exception as e:
        log(f"Cannot connect to Database: {e}", "ERROR")
        log("Check if your IP address is whitelisted in MongoDB Atlas 'Network Access' tab.", "INFO")
        log("Also check if your password in .env is correct.", "INFO")
        return

    log("Starting ParkiSense Early Detection System...", "SUCCESS")
    log("Open your browser at: http://127.0.0.1:5000", "INFO")
    
    try:

        subprocess.run(["python", "main.py"], check=True)
    except KeyboardInterrupt:
        log("Server stopped by user.", "INFO")

if __name__ == "__main__":
    check_dependencies()
    run_app()
