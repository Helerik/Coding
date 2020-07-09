
from flask import Flask, render_template

app = Flask(__name__)

# An example
@app.route('/')
@app.route("/profile/<name>")
def profile(name = None):
    return render_template("profile.html", name = name)

# Starts the web-server
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True)
