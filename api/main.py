from flask import Flask
from flask import request, redirect
from flask_httpauth import HTTPBasicAuth
from database import DB

app = Flask(__name__)

auth = HTTPBasicAuth()

db = DB()

@app.route('/register', methods=['POST'])
def register():
    pass #TODO
    print("REGISTER ATTEMPT!")
    print(request.form['username'])
    return {"success": True}

@app.route('/login', methods=['PUT'])
def login():
    pass #TODO
    return redirect('/')

@app.route('/favourites')
@auth.login_required
def favourites_redirect():
    pass #TODO

@app.route('/favourites/<uid>')
def favourites(uid):
    return db.get_favourites(uid)

if __name__ == '__main__':
    app.run()
