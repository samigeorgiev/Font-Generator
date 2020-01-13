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
        return {"success":False, "status":409, "error_message":"username already taken"}
    if (len(password) < 8):
        return {"success":False, "status":400, "error_message":"password too short"}
    if (len(password) > 64):
        return {"success":False, "status":400, "error_message":"password too long"}
    success = db.add_new_user(username, password)
    if success:
        print("REGISTERED NEW USER!")
        return {"success": True, "status":200}
    return {"success":False, "status":500, "error_message":"internal server error"}

@app.route('/login', methods=['POST'])
def login():
    # TODO
    return {"success": True, "status":200}

@app.route('/favourites/<uid>')
def favourites(uid):
    # TODO
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
