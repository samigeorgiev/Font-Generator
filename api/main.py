from flask import Flask
from flask import request, redirect, jsonify
import database as db

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
        return jsonify({"success": True), 200
    return jsonify({"success":False, "error_message":"internal server error"}), 500

@app.route('/user_exists', methods=['GET'])
def user_exists():
    username = request.form['username']
    return {"success":True, "status":200, "response":db.username_exists(username)}

@app.route('/login', methods=['POST'])
def login():
    # TODO
    return jsonify({"success": True}), 200

@app.route('/favourites/<uid>')
def favourites(uid):
    # TODO
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
