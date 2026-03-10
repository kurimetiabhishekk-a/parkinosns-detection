import os
from pymongo import MongoClient
from dotenv import load_dotenv
import time

def cleanup_database():
    # Load the updated .env (make sure you updated this first!)
    load_dotenv()
    
    uri = os.environ.get('MONGODB_URI')
    
    if not uri or '<password>' in uri or uri == '':
        print("\n❌ STOP!")
        print("You haven't updated your '.env' file yet.")
        print("Please open the file '.env' and paste your new MongoDB link there.")
        return

    try:
        print("\n🔄 Connecting to your secure MongoDB...")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client['parkisense']
        
        # Check if users collection exists
        if 'users' in db.list_collection_names():
            print("🕒 Wiping old insecure data in 3 seconds...")
            time.sleep(3)
            db.users.drop()
            print("✅ DONE! Your database is now fresh and SECURE.")
            print("🚀 You can now go to the Login page and Register a new account safely.")
        else:
            print("ℹ️ Your database is already clean and ready to use!")
            
    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("\nCommon fixes for beginners:")
        print("1. Did you paste the new link correctly in '.env'?")
        print("2. Did you whitelist your IP in MongoDB Atlas (Network Access tab)?")

if __name__ == "__main__":
    cleanup_database()
