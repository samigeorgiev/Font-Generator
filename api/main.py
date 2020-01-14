from secrets import jwt_secret
from flask import Flask
from flask import request, redirect, jsonify
import database as db
import datetime
import jwt

app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    print("REGISTER ATTEMPT!")
    print(username)
    if (db.username_exists(username)):
        return jsonify({"success":False, "error_message":"username already taken"}), 409
    if (len(password) < 8):
        return jsonify({"success":False, "error_message":"password too short"}), 400
    if (len(password) > 64):
        return jsonify({"success":False, "error_message":"password too long"}), 400
    success = db.add_new_user(username, password)
    if success:
        print("REGISTERED NEW USER!")
        return jsonify({"success":True}), 200
    return jsonify({"success":False, "error_message":"internal server error"}), 500

@app.route('/user_exists', methods=['GET'])
def user_exists():
    username = request.form['username']
    return {"success":True, "status":200, "response":db.username_exists(username)}

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if (not db.check_credentials(username, password)):
        return jsonify({"success":False, "error_message":"username or password not correct"}), 401
    token = encode_jwt(db.get_uid(username))
    return jsonify({"success": True, "token":token}), 200

def encode_jwt(uid):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
        'iat': datetime.datetime.utcnow(),
        'sub': uid
    }
    return jwt.encode(
        payload,
        jwt_secret,
        algorithm='HS256'
    ).decode('utf-8')

def decode_jwt(jwt):
    payload = jwt.decode(jwt.encode('utf-8'), jwt_secret)
    return payload['sub']

@app.route('/favourites/<uid>')
def favourites(uid):
    # TODO
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
