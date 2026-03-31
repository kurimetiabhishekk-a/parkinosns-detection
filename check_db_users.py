import sqlite3
import os

db_path = 'mydatabase.db'

if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
else:
    try:
        con = sqlite3.connect(db_path)
        cursor = con.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Users';")
        if not cursor.fetchone():
             print("Users table does not exist.")
        else:
            cursor.execute("SELECT Name, Email, password FROM Users")
            rows = cursor.fetchall()
            if not rows:
                print("No users found in Users table.")
            else:
                print("Users found:")
                for row in rows:
                    print(f"Name: {row[0]}, Email: {row[1]}, Password: {row[2]}")
        con.close()
    except Exception as e:
        print(f"Error reading database: {e}")
