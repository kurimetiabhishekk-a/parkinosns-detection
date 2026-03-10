import os
from dotenv import load_dotenv, set_key
from cryptography.fernet import Fernet
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

ENV_PATH = '.env'

def setup_security():
    # 1. Generate a valid key
    new_key = Fernet.generate_key().decode()
    print(f"Generated Key: {new_key} (len={len(new_key)})")
    
    # 2. Update .env
    load_dotenv(ENV_PATH)
    os.environ['ENCRYPTION_KEY'] = new_key
    with open(ENV_PATH, 'r') as f:
        content = f.read().splitlines()
    
    new_content = []
    found = False
    for line in content:
        if line.startswith('ENCRYPTION_KEY='):
            new_content.append(f'ENCRYPTION_KEY={new_key}')
            found = True
        else:
            new_content.append(line)
    if not found:
        new_content.append(f'ENCRYPTION_KEY={new_key}')
    
    with open(ENV_PATH, 'w') as f:
        f.write('\n'.join(new_content) + '\n')
    
    print("Updated .env file.")

    # 3. Seed demo user
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("Error: MONGODB_URI not found in environment.")
        return

    try:
        client = MongoClient(uri)
        db = client['parkisense']
        f = Fernet(new_key.encode())
        
        db.users.delete_many({})
        db.users.insert_one({
            'email': 'demo@parkisense.com',
            'password': generate_password_hash('demo1234'),
            'name': f.encrypt('Demo User'.encode()).decode(),
            'pet': f.encrypt('buddy'.encode()).decode(),
            'date': '08/03/2026 00:00:00'
        })
        print("Successfully seeded demo user: demo@parkisense.com / demo1234")
    except Exception as e:
        print(f"Error seeding user: {e}")

if __name__ == "__main__":
    setup_security()
