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

old_email = 'abhishekkueimeti97@gmail.com'
new_email = 'abhishekkurimeti97@gmail.com'
correct_pet = 'toto'

print(f"Searching for {old_email}...")
user = db.users.find_one({'email': old_email})

if user:
    print("User found. Updating email and pet name...")
    db.users.update_one(
        {'_id': user['_id']},
        {
            '$set': {
                'email': new_email,
                'pet': fernet.encrypt(correct_pet.encode()).decode(),
                'name': fernet.encrypt('Abhishek'.encode()).decode() # Assuming name or just keeping it
            }
        }
    )
    print("Update successful!")
else:
    print(f"User {old_email} not found. Checking for the correct one just in case...")
    user = db.users.find_one({'email': new_email})
    if user:
        print("User found with correct email. Updating pet name...")
        db.users.update_one(
            {'_id': user['_id']},
            {
                '$set': {
                    'pet': fernet.encrypt(correct_pet.encode()).decode()
                }
            }
        )
        print("Update successful!")
    else:
        print("User not found at all. Creating fresh...")
        from werkzeug.security import generate_password_hash
        import datetime
        db.users.insert_one({
            'date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'name': fernet.encrypt('Abhishek'.encode()).decode(),
            'email': new_email,
            'password': generate_password_hash('demo1234'), # Setting a default password if they are stuck
            'pet': fernet.encrypt(correct_pet.encode()).decode()
        })
        print("Fresh account created for abhishekkurimeti97@gmail.com")
