from flask import Flask
from flask import request, redirect
from flask_httpauth import HTTPBasicAuth
import database as db

app = Flask(__name__)

auth = HTTPBasicAuth()

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    print("REGISTER ATTEMPT!")
    print(username)
    if (db.username_exists(username)):
        return {"success":False, "error":"username already taken"}
    if (len(password) < 8):
        return {"success":False, "error":"password too short"}
    if (len(password) > 64):
        return {"success":False, "error":"password too long"}
    success = db.add_new_user(username, password)
    if success:
        print("REGISTERED NEW USER!")
        return {"success": True}
    return {"success":False, "error":"internal server error"}

@app.route('/login', methods=['POST'])
def login():
    # TODO
    return {"success": True}

@app.route('/favourites/<uid>')
def favourites(uid):
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
