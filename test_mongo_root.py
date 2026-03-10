import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
uri = os.environ.get('MONGODB_URI')
# Remove /parkisense if present to see if we can at least authenticate to the cluster
if '?' in uri:
    parts = uri.split('?')
    base = parts[0]
    if base.count('/') >= 3:
        new_base = "/".join(base.split('/')[:3]) + "/"
        uri = new_base + "?" + parts[1]

print(f"Testing root authentication with URI: {uri}")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("SUCCESS: Root authentication works!")
except Exception as e:
    print(f"FAILED: {e}")
