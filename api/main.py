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
    data = json.loads(request.data)
    email = data['email']
    username = data['name']
    password = data['password']
    print(username)
    print(email)
    if (db.email_exists(email)):
        return jsonify({"success":False, "error_message":"email already registered"}), 409
    if (db.username_exists(username)):
        return jsonify({"success":False, "error_message":"username already taken"}), 409
    if (len(password) < 8):
        return jsonify({"success":False, "error_message":"password too short"}), 400
    if (len(password) > 64):
        return jsonify({"success":False, "error_message":"password too long"}), 400
    success = db.add_new_user(email, username, password)
    if success:
        print("REGISTERED NEW USER!")
        return jsonify({"success":True}), 201
    return jsonify({"success":False, "error_message":"internal server error"}), 500

@app.route('/api/user_exists', methods=['GET'])
def user_exists():
    data = json.loads(request.data)
    email = data['email']
    return jsonify({"success":True, "response":db.email_exists(email)}), 200

@app.route('/api/login', methods=['POST'])
def login():
    data = json.loads(request.data)
    email = data['email']
    password = data['password']
    if (not db.check_credentials(email, password)):
        return jsonify({"success":False, "error_message":"email or password not correct"}), 401
    return encode_jwt(db.get_uid(email))

def encode_jwt(uid):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
        'iat': datetime.datetime.utcnow(),
        'sub': uid
    }
    return jsonify({
        "success": True,
        "exp":payload["exp"],
        "token":jwt.encode(
            payload,
            jwt_secret,
            algorithm='HS256'
        ).decode('utf-8')
    }), 200

def decode_jwt(token):
    payload = jwt.decode(token.encode('utf-8'), jwt_secret)
    return payload['sub']

@app.route('/api/favourites/<uid>')
def favourites(uid):
    # TODO
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
