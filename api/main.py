# pylint: disable=no-member

import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from secrets import jwt_secret

import database as db
import jwt
from flask import Flask, jsonify, redirect, request
from flask_cors import CORS

from neural_api import get_pair_by_contrast


app = Flask(__name__)
CORS(app)

@app.route('/api/register', methods=['POST'])
def register():
    data = json.loads(request.data)
    email = data['email']
    username = data['name']
    password = data['password']

    try:
        app.logger.debug("Registration attempt ('%s', '%s')" % (username, email))

        if (db.email_exists(email)):
            app.logger.debug("Registration for '%s' failed - email already taken" % email)
            return jsonify({"success":False, "error_message":"email already registered"}), 409

        if (db.username_exists(username)):
            app.logger.debug("Registration for '%s' failed - username already taken" % username)
            return jsonify({"success":False, "error_message":"username already taken"}), 409

        if (len(password) < 8):
            app.logger.debug("Registration for '%s' failed - password too short" % username)
            return jsonify({"success":False, "error_message":"password too short"}), 422

        if (len(password) > 64):
            app.logger.debug("Registration for '%s' failed - password too long" % username)
            return jsonify({"success":False, "error_message":"password too long"}), 422

        db.add_new_user(email, username, password)
        app.logger.info("Registered new user ('%s', '%s')" % (username, email))

        return jsonify({"success":True}), 201

    except Exception as e:
        app.logger.error("Error while processing registration ('%s', '%s')\n%s" % (username, email, e))
        return jsonify({"success":False, "error_message":"internal server error"}), 500

@app.route('/api/user_exists', methods=['GET'])
def user_exists():
    data = json.loads(request.data)
    email = data['email']
    app.logger.debug("Check if email '%s' exists" % email)
    return jsonify({"success":True, "response":db.email_exists(email)}), 200

@app.route('/api/login', methods=['POST'])
def login():
    data = json.loads(request.data)
    email = data['email']
    password = data['password']
    app.logger.debug("Attempt to login as '%s'" % email)
    if (not db.check_credentials(email, password)):
        app.logger.debug("Invalid credentials on login as '%s'" % email)
        return jsonify({"success":False, "error_message":"email or password not correct"}), 401
    app.logger.info("User '%s' logged in" % email)
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

if jwt_secret == "[temporary]":
    print("WARNING: Please replace the jwt secret in secrets.py to a real secret.")

#################### Neural Routing ####################

@app.route('/api/new-font', methods=['POST'])
def new_font():
    data = json.loads(request.data)

    heading = get_pair_by_contrast(
        data['fonts']['heading'], data['deltaContrast'])

    body = get_pair_by_contrast(
        heading, data['deltaContrast'])

    return jsonify({
        'heading': heading,
        'body': body,
    })

@app.route('/api/recommend', methods=['GET'])
def recommend():
    return jsonify({
        'heading': 'Amarante',
        'body': 'Amarante',
    }), 200

########################################################

@app.route('/api/save-font', methods=['POST'])
def save_font():
    # TODO - logging
    data = json.loads(request.data)
    fonts = data['fonts']
    token = request.headers['Authorization']
    print(token)
    try:
        uid = decode_jwt(token)
        db.save_font(uid, fonts)
        return jsonify({"success":True}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"success":False, "error_message":"expired token"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success":False, "error_message":"invalid token"}), 400
    except Exception as e:
        print(e)
        return jsonify({"success":False, "error_message":"internal server error"}), 500

if __name__ == '__main__':
    today = datetime.date.today().strftime("%d%m%y")
    handler = RotatingFileHandler(
        "%s.log" % today,
        maxBytes = 10000,
        backupCount = 1
    )
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.run()
