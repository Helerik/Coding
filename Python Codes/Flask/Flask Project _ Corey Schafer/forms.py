
# Creates HTML forms from python using WTForms
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo

# DataRequired makes so that it can't bre empty.
class RegistrationForm(FlaskForm):
    username = StringField("Username",
                           validators = [DataRequired(), Length(min = 3, max = 20)])
    email = StringField("Email",
                        validators = [DataRequired(), Email()])
    password = PasswordField("Password",
                             validators = [DataRequired()])
    confirm_password =  PasswordField("Confirm Password",
                                      validators = [DataRequired(), EqualTo("password")])
    submit = SubmitField("Sign Up")

class LoginForm(FlaskForm):
    email = StringField("Email",
                        validators = [DataRequired(), Email()])
    password = PasswordField("Password",
                             validators = [DataRequired()])
    remember = BooleanField("Remember Me")
    submit = SubmitField("Login")
