
from flask import Flask, render_template, url_for, redirect, flash
from forms import RegistrationForm, LoginForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "78b043556da3d624dcade05493d244ff"

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

@app.route("/register", methods = ["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}!", "success")
        return redirect(url_for("home_page"))
    return render_template("register.html",
                           title = "Register",
                           form = form)

@app.route("/login", methods = ["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Don't have a database for registering users yet
        return redirect(url_for("home_page"))
    else:
        flash("Login Unsuccessful. Check your username and password", "danger")
    return render_template("login.html",
                           title = "Login",
                           form = form)















if __name__ == "__main__":
    app.run(debug = True)
