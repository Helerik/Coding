
from flask import Flask, request

app = Flask(__name__)

@app.route('/') #root directory, i.e. homepage
def index():
    return "This is the homepage"

@app.route("/some_name")
def some_name():
    return "<h2>This is a web page text in HTML</h2>"

@app.route("/profile/<username>") # <username> is a variable
def profile(username):
    return f"Hey there {username}."

@app.route("/post/<int:post_id>") # if datatype != string, has to pass data type!
def post(post_id):
    return f"<h2>Post ID is {post_id}<h2>"

# About requests
@app.route("/new_page")
def new_page():
    return f"Method used: {request.method}"

@app.route("/bacon", methods = ["GET", "POST"])
def bacon():
    if request.method == "POST":
        return "Using POST"
    else:
        return "Probably using GET"

# Starts the web-server
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True)
