from secrets import jwt_secret
from flask import Flask
from flask import request, redirect, jsonify
from flask_cors import CORS
import database as db
import datetime
import jwt
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/register', methods=['POST'])
def register():
    print("REGISTER ATTEMPT!")
    data = json.loads(data)
    print(list(args.keys()))
    username = data['email']
    password = data['password']
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

@app.route('/api/user_exists', methods=['GET'])
def user_exists():
    data = json.loads(request.data)
    username = data['email']
    return {"success":True, "status":200, "response":db.username_exists(username)}

@app.route('/api/login', methods=['POST'])
def login():
    data = json.loads(request.data)
    username = data['email']
    password = data['password']
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

def decode_jwt(token):
    payload = jwt.decode(token.encode('utf-8'), jwt_secret)
    return payload['sub']

@app.route('/api/favourites/<uid>')
def favourites(uid):
    # TODO
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
