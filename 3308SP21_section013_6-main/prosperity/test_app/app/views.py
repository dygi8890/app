from flask import render_template
from app import app


@app.route("/")
def home():
    return render_template("home.html")

def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'images/fabio.jpg')
    return render_template("index.html", user_image = full_filename)


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/account_reg")
def account_reg():
    return render_template("account_reg.html")