import os
from dotenv import load_dotenv
from pymongo import MongoClient
from cryptography.fernet import Fernet

load_dotenv()
uri = os.environ.get('MONGODB_URI', '').strip()
enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
fernet = Fernet(enc_key.encode())

client = MongoClient(uri)
db = client['parkisense']

email = 'abhishekkurimeti97@gmail.com'
print(f"Fixing name for {email}...")

user = db.users.find_one({'email': email})
if user:
    # Re-encrypt name with the CORRECT current key
    new_name = fernet.encrypt("Abhishek".encode()).decode()
    db.users.update_one(
        {'_id': user['_id']},
        {'$set': {'name': new_name}}
    )
    print("Name fixed successfully!")
else:
    print("User not found.")
