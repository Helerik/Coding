
from flask import Flask, render_template, url_for

app = Flask(__name__)

posts = [
    {"author": "Erik Vincent",
     "date": "07/10/2020",
     "title": "The posts",
     "number": "1",
     "content": "Blah foo fun"},
    
    {"author": "Corey Schafer",
     "date": "04/05/2018",
     "title": "Python Flask Tutorial",
     "number": "0",
     "content": "I will teach you how to use Flask"}
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
