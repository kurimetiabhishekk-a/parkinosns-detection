import os
from pymongo import MongoClient
from dotenv import load_dotenv
from cryptography.fernet import Fernet

def verify():
    load_dotenv()
    uri = os.environ.get('MONGODB_URI')
    key = os.environ.get('ENCRYPTION_KEY')
    
    print(f"--- Environment Check ---")
    print(f"MONGODB_URI exists: {bool(uri)}")
    print(f"ENCRYPTION_KEY exists: {bool(key)}")
    if key:
        print(f"Key length: {len(key.strip())}")
    
    try:
        client = MongoClient(uri)
        db = client['parkisense']
        users = list(db.users.find())
        print(f"\n--- Database Check ---")
        print(f"Total Users: {len(users)}")
        
        if users:
            f = Fernet(key.encode().strip())
            for u in users:
                print(f"\nUser Email: {u.get('email')}")
                print(f"Raw Name in DB: {u.get('name')}")
                try:
                    decrypted_name = f.decrypt(u.get('name').encode()).decode()
                    print(f"Decrypted Name: {decrypted_name}")
                except Exception as e:
                    print(f"Decryption FAILED for Name: {e}")
                
                print(f"Password Hash (start): {u.get('password')[:20]}...")
        else:
            print("No users found in database.")
            
    except Exception as e:
        print(f"Connection/Verification Error: {e}")

if __name__ == "__main__":
    verify()
