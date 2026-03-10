import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash
from cryptography.fernet import Fernet
from pymongo import MongoClient

load_dotenv()
uri = os.environ.get('MONGODB_URI', '').strip()
enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
fernet = Fernet(enc_key.encode())

def encrypt_data(data):
    return fernet.encrypt(data.encode()).decode()

print("Connecting to MongoDB...")
client = MongoClient(uri, serverSelectionTimeoutMS=10000)
db = client['parkisense']

print("Dropping existing users to fix login issues...")
db.users.drop()

print("Creating fresh demo account...")
import datetime
db.users.insert_one({
    'date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    'name': encrypt_data('Demo User'),
    'email': 'demo@parkisense.com',
    'password': generate_password_hash('demo1234'),
    'pet': encrypt_data('buddy')
})

print("Demo account created!")
print("Email: demo@parkisense.com")
print("Password: demo1234")
