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

print("All users in DB:")
for user in db.users.find():
    email = user.get('email')
    pet_enc = user.get('pet')
    pet_dec = "ERROR"
    if pet_enc:
        try:
            pet_dec = fernet.decrypt(pet_enc.encode()).decode()
        except:
            pet_dec = "[DECRYPTION FAILED]"
    
    print(f"Email: '{email}' | Decrypted Pet: '{pet_dec}'")
