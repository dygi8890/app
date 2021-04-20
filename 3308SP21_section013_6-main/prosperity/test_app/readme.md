To build Flask app in Docker container, go to directory /test_app and:

1) Check that the port 56733 is free with:
	sudo nc localhost 56733 < /dev/null; echo $?
   If it's not free, edit start.sh to contain a free port with:
	sudo nano start.sh

2) Run Docker image:

	a) To start the image up after it's been built:
		docker start photophonic
	
	a) If the docker container isn't built at all, run:
		sudo bash start.sh

3) Check that port is occupied by Docker container with:
	sudo docker ps

4) Go to  http://your-local-ip-address:56733 to visit app!

-----------------------------------------------------------------------

5) To update app with a container restart, run:
	sudo docker stop photophonic && sudo docker start photophonic

6) To close the app, run:
	docker stop photophonic


To test PostgreSQL run:
docker exec -it CONTAINER_ID psql -U postgres postgres
