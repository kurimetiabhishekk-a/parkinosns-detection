import sys
import itertools
from pymongo import MongoClient
import urllib.parse
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def try_conn(pwd):
    encoded_pwd = urllib.parse.quote_plus(pwd)
    uri = f"mongodb+srv://abhishekkurimeti97_db_user:{encoded_pwd}@cluster0.gege15n.mongodb.net/parkisense?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tlsAllowInvalidCertificates=True)
        client.admin.command('ping')
        return True
    except Exception as e:
        return False

base_raw = "E1lOIPImeOw43Izw"
base = list(base_raw)

variants = {
    1: ['1', 'l', 'I'],
    2: ['l', '1', 'I'],
    3: ['O', '0'],
    4: ['I', 'l', '1'],
    6: ['I', 'l', '1'],
    9: ['O', '0'],
    13: ['I', 'l', '1']
}

keys = sorted(variants.keys())
options = [variants[k] for k in keys]

print(f"Testing {len(list(itertools.product(*options)))} combos...")

for combo in itertools.product(*options):
    p = list(base)
    for i, char in enumerate(combo):
        p[keys[i]] = char
    pwd = "".join(p)
    if try_conn(pwd):
        print(f"SUCCESS! Correct PWD: {pwd}")

        with open("found_pwd.txt", "w") as f:
            f.write(pwd)
        sys.exit(0)

print("Failed to find password.")
