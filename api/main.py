from flask import Flask
from flask import render_template
from database import DB

app = Flask(__name__)

auth = HTTPBasicAuth()

db = DB()

@app.route('/')
def home_page():
    return render_template('home_page.html')

@app.route('/login', methods=['GET', 'POST', 'PUT'])
def login_page():
    if request.method == 'GET': # just got to the page
        return render_template('login_page.html')
    elif request.method == 'PUT': # wants to login
        pass #TODO
        return redirect('/')
    else: # wants to register from the login page
        pass #TODO 
        return redirect('/')

@app.route('/favourites')
@auth.login_required
def favourites_redirect():
    pass #TODO

@app.route('/favourites/<uid>')
def favourites(uid):
    return render_template('favourites.html', posts=db.get_favourites(uid))

if __name__ == '__main__':
    app.run()
