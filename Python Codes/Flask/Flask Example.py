
from flask import Flask

app = Flask(__name__)

@app.route('/') #root directory, i.e. homepage
def index():
    return "This is the homepage"

@app.route('/some_name')
def some_name():
    return '<h2>This is a web page text in HTML</h2>'

# Starts the web-server
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True)
