import sqlite3

db = sqlite3.connect('database.sqlite3')

db.cursor().execute('''
CREATE TABLE IF NOT EXISTS `users` (
    `user_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `username` TEXT NOT NULL UNIQUE
)''')
db.cursor().execute('''
CREATE TABLE IF NOT EXISTS `bookmarks` (
    `bookmark_id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `user_id` INTEGER,
    `preview` TEXT,
    `font` TEXT ,
    FOREIGN KEY(`user_id`) REFERENCES `users`(`userid`)
)''')
db.commit()

class DB:
    def __enter__(self):
        self.conn = sqlite3.connect(DB_NAME)
        return self.conn.cursor()

    def __exit__(self, type, value, traceback):
        self.conn.commit()
