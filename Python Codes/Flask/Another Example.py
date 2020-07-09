
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route("/<user>")
def index(user = None):
    return render_template("user.html", user = user)

@app.route("/shopping")
def shopping():
    food = ["cheese", "tuna", "bacon", "strawberry"]
    return render_template("shopping.html", food = food)

if __name__ == "__main__":
    app.run(debug = True)
