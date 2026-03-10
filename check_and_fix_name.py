import os
from pymongo import MongoClient
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# Hardcoded key from main.py
KEY = "opVjAjT3z__mi9-j0dWS6idv5GqHFuk7CFQvuwB5Gio="
URI = "mongodb+srv://abhishekkurimeti97_db_user:ParkiSense_2026_Secure123@cluster0.gege15n.mongodb.net/parkisense?retryWrites=true&w=majority"

fernet = Fernet(KEY.encode())
client = MongoClient(URI)
db = client['parkisense']

email = 'abhishekkurimeti97@gmail.com'
user = db.users.find_one({'email': email})

print(f"Checking user: {email}")
if user:
    encrypted_name = user.get('name')
    print(f"Encrypted Name in DB: {encrypted_name}")
    try:
        decrypted = fernet.decrypt(encrypted_name.encode()).decode()
        print(f"Decrypted Result: {decrypted}")
    except Exception as e:
        print(f"Decryption FAILED: {e}")
        # Try to re-encrypt properly right now
        print("Attempting to re-encrypt with current key...")
        new_encrypted = fernet.encrypt("Abhishek".encode()).decode()
        db.users.update_one({'_id': user['_id']}, {'$set': {'name': new_encrypted}})
        print("Re-encryption complete.")
else:
    print("User not found.")
