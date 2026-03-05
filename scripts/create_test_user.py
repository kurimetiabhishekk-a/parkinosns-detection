import sqlite3
from datetime import datetime

db = 'mydatabase.db'
email = 'test@local'
name = 'Test User'
password = 'password123'
pet = 'fluffy'

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS Users (Date text,Name text,Email text,password text,pet text)")
# check existing
cur.execute("SELECT COUNT(*) FROM Users WHERE Email=?", (email,))
if cur.fetchone()[0] == 0:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cur.execute("INSERT INTO Users VALUES(?,?,?,?,?)", (now, name, email, password, pet))
    con.commit()
    print('Inserted user:', email)
else:
    print('User already exists:', email)

cur.execute("SELECT Date,Name,Email,pet FROM Users WHERE Email=?", (email,))
rows = cur.fetchall()
for r in rows:
    print(r)

con.close()
