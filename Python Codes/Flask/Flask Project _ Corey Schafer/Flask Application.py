
from flask import Flask, render_template

app = Flask(__name__)

posts = [
    {"author": "Erik Vincent",
     "date": "07/10/2020",
     "title": "The posts",
     "number": "1",
     "content": "Blah foo fun"}
    ]

@app.route('/')
@app.route("/home")
def home_page():
    return render_template("home_page.html",
                           title = "Home",
                           posts = posts)

@app.route("/about")
def about():
    return render_template("about.html",
                           title = "About")

















if __name__ == "__main__":
    app.run(debug = True)
