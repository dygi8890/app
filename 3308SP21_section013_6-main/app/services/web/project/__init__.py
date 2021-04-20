
#------------------------------------------------
#
#
# Author: The Photophonic Team
# CSCI_3308, Photophonic Web-App Project
# Created on 3/8/21
#
#
#------------------------------------------------


from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
import os
from os import listdir
import photophonic as pp # main audio generation and image processing definitions


app = Flask(__name__)
app.config.from_object("project.config.Config")
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://hello_flask:hello_flask@db:5432/hello_flask_dev"
db = SQLAlchemy(app)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static/uuids'),
    DEBUG = True,
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
)
app.secret_key = "temporary_secret_key"

# test1 = db.Table('users', db.metadata, autoload=True, autoload_with=db.engine)

# class User(db.Model):
#     __tablename__ = "users"
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(128), unique=True, nullable=False)
#     active = db.Column(db.Boolean(), default=True, nullable=False)
#     def __init__(self, email):
#         self.email = email

# Map to existing database
Base = automap_base()
Base.prepare(db.engine, reflect=True)
UsersDb = Base.classes.users
ImagesDb = Base.classes.images


@app.route('/', methods = ['GET'])
def home():

    if request.args: # if a uuid has been supplied
        uuid = request.args.get('uuid')

    else:
        uuid = 'NGGYU'  # default audio uuid

    print("\n[DEBUG]: ImagesDb:")
    for class_instance in db.session.query(ImagesDb).all():
        image_id = vars(class_instance)["image_id"]
        user_name = vars(class_instance)["user_name"]
        print("\'image_id\': {} \'user_name\': {}".format(image_id, user_name))
    print()

    return render_template("home.html", uuid=uuid)


@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if f != None:

            image_id, image_matrix = pp.makeUUID(f, app.config['UPLOADED_PATH']) # turns image into uuid-named image and audio files

            # -------------------------------------------------------------#
            # Insert image to DB here!: (image_id, image_matrix, username)
            #
            # id            => uuid for image (universally unique identifier)
            # image_matrix  => list containing pixel data
            # username      => ...?
            # -------------------------------------------------------------#

            user_name = "Bob The Third"

            newImage = ImagesDb(image_id=image_id, user_name=user_name)
            db.session.add(newImage)
            db.session.commit()

            return redirect(url_for('home', uuid=image_id))


# @app.route('/load')
# def load(): # point here when selecting an image in the user's gallery or the explore tab

    # -----------------------------------------------------------------------------------------#
    # 1) Load image with ID==UUID FROM DB (save image to server directory static/uuids/)
    # 2) Get image matrix with DB call: "FROM images SELECT image_matrix WHERE ID = uuid"
    # 3) Write image matrix to server with cv2.imwrite(image_matrix, PATH/uuid.jpg)
    # -----------------------------------------------------------------------------------------#


@app.route("/login", methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':

        # get inputed username and password
        inputUser = request.form['username']
        inputPassword = request.form['password']

        # query database
        userResult = db.session.query(UsersDb).filter_by(user_name=inputUser).first()
        if userResult:
            passResult = (userResult.password == inputPassword)
        else:
            passResult = False

        if not userResult or not passResult:
            error = 'Invalid Credentials. Please try again.'
        else:
            flash('Log in successful!')
            return redirect("/")
    return render_template("login.html", error=error)


@app.route("/account_reg", methods = ['GET','POST'])
def account_reg():
    error = None
    if request.method == 'POST':

        newUsername = request.form['newUsername']
        newEmail = request.form['newEmail']
        newPassword = request.form['newPassword']

        checkUsername = db.session.query(UsersDb).filter_by(user_name=newUsername).first()
        if checkUsername:
            error = 'Username already Exists.'
        elif(newUsername and newEmail and newPassword):
            new_user = UsersDb(user_name = newUsername, password = newPassword)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully!')
            return redirect("/")
    return render_template("account_reg.html", error=error)

@app.route("/creations")
def creations():

    rawUuids = listdir(app.config['UPLOADED_PATH'])
    uuids = []
    for uuid in rawUuids:
        print(uuids)
        split=uuid.split('.')
        if split[1] != "jpg" and split[0] not in uuids:
            uuids.append(split[0])
    uuids.remove('')

    half = len(uuids) // 2
    listOneUuids = uuids[:half]
    listTwoUuids = uuids[half:]

    listOne = '<div class="d-flex flex-row flex-nowrap overflow-auto" style="height:20vw;">'
    listTwo = listOne

    for uuid in listOneUuids:
        listTemplate = '<div class="card card-block mx-2" style="min-width: 20vw; max-width: 30vw;"><img class="d-block w-100" style="width:100%; height:100%;" src="/static/uuids/' + uuid + '.jpg" alt="' + uuid + '"></div>'
        listOne = listOne + listTemplate
    listOne = listOne + '</div>'

    for uuid in listTwoUuids:
        listTemplate = '<div class="card card-block mx-2" style="min-width: 20vw; max-width: 30vw;"><img class="d-block w-100" style="width:100%; height:100%;" src="/static/uuids/' + uuid + '.jpg" alt="' + uuid + '"></div>'
        listTwo = listTwo + listTemplate
    listTwo = listTwo + '</div>'

    return render_template("creations.html", uuids=uuids, path=app.config['UPLOADED_PATH'], listOne=listOne, listTwo=listTwo)


# @app.route('/test',methods = ['POST', 'GET'])
# def test():
#     return render_template("test.html")
#
#
# @app.route('/result', methods=['POST', 'GET'])
# def result():
#     if request.method == 'POST':
#         if request.form['submit_button'] == 'Hedgehog':
#             filename = pp.colorMark('hedgehog', '.jpeg')
#             return render_template(     "result.html",
#                                         image='hedgehog.jpeg',
#                                         dimensions=pp.getImageDimensions('hedgehog.jpeg'),
#                                         markedImage=filename
#                                     )
#         elif request.form['submit_button'] == 'Cloud':
#             filename = pp.colorMark('cloud', '.jpg')
#             return render_template(     "result.html",
#                                         image='cloud.jpg',
#                                         dimensions=pp.getImageDimensions('cloud.jpg'),
#                                         markedImage=filename
#                                    )
#         else:
#             pass  # unknown
#     elif request.method == 'GET':
#         return render_template('result.html')

