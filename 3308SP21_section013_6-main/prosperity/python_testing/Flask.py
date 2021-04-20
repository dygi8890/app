from flask import Flask, render_template

app = Flask(__name__)


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/account_reg")
def account_reg():
    return render_template("account_reg.html")


if __name__ == "__main__":
    app.run()
