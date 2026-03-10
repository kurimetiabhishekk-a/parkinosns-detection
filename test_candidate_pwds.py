from pymongo import MongoClient
import os

passwords = [
    "Jes58riI1e3QgAit",
    "E1lOIPImeOw43Izw",
    "gA4TIfnkmZMGY3G4",
    "E1lOIPlmeOw43Izw",
    "zV34o7w3MBuT1BQRhkhiBmYvopvO6rhRECUktG300Nc="
]

for p in passwords:
    uri = f"mongodb+srv://abhishekkurimeti97_db_user:{p}@cluster0.gege15n.mongodb.net/parkisense?retryWrites=true&w=majority"
    print(f"Testing password: {p}")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print(f"SUCCESS! Found correct password: {p}")
        exit(0)
    except Exception as e:
        print(f"Failed: {e}")

print("None worked.")
