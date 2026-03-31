
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.environ.get('MONGODB_URI', '').strip()
enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()

results = []
results.append(f"MongoDB URI found: {bool(uri)}")
results.append(f"Encryption key found: {bool(enc_key)}")

from cryptography.fernet import Fernet
try:
    fernet = Fernet(enc_key.encode())
    test_enc = fernet.encrypt(b"test_data")
    test_dec = fernet.decrypt(test_enc)
    results.append(f"Encryption OK")
except Exception as e:
    results.append(f"Encryption ERROR: {e}")

from pymongo import MongoClient
try:
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command('ping')
    results.append("MongoDB: CONNECTED OK")
    
    db = client['parkisense']
    users = list(db.users.find({}, {'email': 1, 'name': 1, 'password': 1, 'pet': 1, '_id': 0}))
    results.append(f"Total users in DB: {len(users)}")
    for u in users[:20]:
        email = u.get('email', 'N/A')
        name = u.get('name', 'N/A')
        has_pwd = bool(u.get('password'))
        pet = u.get('pet', 'N/A')
        name_encrypted = name.startswith('gAAAAA') if isinstance(name, str) else False
        pet_encrypted = pet.startswith('gAAAAA') if isinstance(pet, str) else False
        results.append(f"  {email} | name_enc={name_encrypted} | pwd={has_pwd} | pet_enc={pet_encrypted}")

    indexes = list(db.users.list_indexes())
    results.append(f"Indexes: {[i['name'] for i in indexes]}")
    
except Exception as e:
    results.append(f"MongoDB ERROR: {e}")

with open('test_db_output.txt', 'w') as f:
    for r in results:
        f.write(r + '\n')
        print(r)
