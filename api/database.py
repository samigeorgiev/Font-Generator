import sqlite3
import hashlib

DB_NAME = 'database.sqlite3'

conn = sqlite3.connect(DB_NAME)

conn.cursor().execute('''
CREATE TABLE IF NOT EXISTS `users` (
    `uid` INTEGER PRIMARY KEY AUTOINCREMENT,
    `email` TEXT NOT NULL UNIQUE,
    `username` TEXT NOT NULL UNIQUE,
    `password` TEXT NOT NULL
)''')
conn.cursor().execute('''
CREATE TABLE IF NOT EXISTS `favourites` (
    `fid` INTEGER PRIMARY KEY AUTOINCREMENT,
    `uid` INTEGER,
    `heading` TEXT,
    `body` TEXT ,
    FOREIGN KEY(`uid`) REFERENCES `users`(`uid`)
)''')
conn.commit()
conn.close()

def sql_select1(query, params):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone()
    return result

def sql_select(query, params):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    return result

def sql_execute(query, params):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.rowcount
    conn.commit()
    return result

def email_exists(email):
    query = """SELECT email FROM users WHERE email = ?"""
    result = sql_select1(query, (email,))
    return bool(result)

def username_exists(username):
    query = """SELECT username FROM users WHERE username = ?"""
    result = sql_select1(query, (username,))
    return bool(result)

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def add_new_user(email, username, password):
    query = """INSERT INTO users (email, username, password) VALUES (?,?,?)"""
    sql_execute(query, (email, username,hash_password(password)))

def check_credentials(email, password):
    query = """SELECT email FROM users WHERE email = ? AND password = ?"""
    result = sql_select1(query, (email,hash_password(password)))
    return bool(result)

def get_uid(email):
    query = """SELECT uid FROM users WHERE email = ?"""
    result = sql_select1(query, (email,))
    return result[0]

def save_font(uid, fonts):
    query = """INSERT INTO favourites (uid, heading, body) VALUES (?,?,?)"""
    sql_execute(query, (uid, fonts['heading'], fonts['body']))

def get_saved_fonts(uid):
    query = """SELECT heading, body, fid FROM favourites WHERE uid = ?"""
    return sql_select(query, (uid,))

def delete_font(uid, fid):
    query = """DELETE FROM favourites WHERE fid = ? AND uid = ?"""
    return sql_execute(query, (fid, uid))
