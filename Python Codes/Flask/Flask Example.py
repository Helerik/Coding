
from flask import flask

app = Flask(__name__)

@app.route('/') #root directory, i.e. homepage
def index():
    return "This is the homepage"
