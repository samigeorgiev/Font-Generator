import sqlite3
import hashlib

DB_NAME = 'database.sqlite3'

conn = sqlite3.connect(DB_NAME)

conn.cursor().execute('''
CREATE TABLE IF NOT EXISTS `users` (
    `uid` INTEGER PRIMARY KEY AUTOINCREMENT,
    `username` TEXT NOT NULL UNIQUE,
    `password` TEXT NOT NULL
)''')
conn.cursor().execute('''
CREATE TABLE IF NOT EXISTS `favourites` (
    `fid` INTEGER PRIMARY KEY AUTOINCREMENT,
    `user_id` INTEGER,
    `preview` TEXT,
    `font` TEXT ,
    FOREIGN KEY(`user_id`) REFERENCES `users`(`userid`)
)''')
conn.commit()

def username_exists(username):
    query = """SELECT username FROM users WHERE username = ?"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    if result:
        return True
    else:
        return False

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def add_new_user(username, password):
    try:
        query = """INSERT INTO users (username, password) VALUES (?,?)"""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(query, (username,hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def get_favourites(uid):
    pass #TODO
    return {"bs":True}

