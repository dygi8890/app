##Fun Fact:
Flask is smart: _Just_ saving the __init__.py file will tell it to rebuild the webapp. No nead to rebuild/restart the container. noice

###Build container and spin up the two containers after changes with:
    $ docker-compose up -d --build

Tada! Might take a hot sec, OpenCV is pretty beefy these days.

Visit http://localhost:5000/ to see the container. Once running...

<br/>

###To see debugging outputs in the terminal (python print() statements), and save yourself from building the project again, just run:
    $ docker-compose restart
####or
    $ docker-compose up

<br/>

###Open the PostgresSQL CLI (command line interface)
    $ docker-compose exec web python manage.py seed_db
    $ docker-compose exec db psql --username=hello_flask --dbname=hello_flask_dev

<br/>

##If you desire to build and start the containers, but _slowly_

####To _just_ re-build the image:
    $ docker-compose build
####Once the image is built, run the container:
    $ docker-compose up -d
Navigate to http://localhost:5000/ again for a sanity check.

####If these don't work, check for errors in the logs via:
    $ docker-compose logs -f

<br/>
<br/>

Following a tutorial on implementing Docker containerization for PostgresSQL and Flask: https://testdriven.io/blog/dockerizing-flask-with-postgres-gunicorn-and-nginx/