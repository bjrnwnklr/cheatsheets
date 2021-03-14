# Docker

Example - running the Random Poetry app as a Docker image

[Miguel Grinberg's Flask Mega tutorial - Docker file](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xix-deployment-on-docker-containers)
[vsupalov - Docker usage in the Flask Mega tutorial](https://vsupalov.com/flask-megatutorial-review/)


# Dockerfile

- download nltk cmudict using `python -m nltk.downloader cmudict` - this will download the cmudict to /nltk_data/.

```Dockerfile
FROM python:3.9.1

# create user so we don't run as root, then change into the homedir of the user and switch to the user
# The rest of the dockerfile will now be run under this user.
RUN adduser --disabled-login randompoetry
WORKDIR /home/randompoetry
USER randompoetry

# create a virtual environment and install the requirements into it
COPY docker-requirements.txt requirements.txt
RUN python -m venv .venv
RUN .venv/bin/pip install -r requirements.txt
RUN .venv/bin/pip install gunicorn

# Download nltk cmudict - latest version. Do this after installing python requirements, as nltk is being installed there.
# The -d option specifies the directory to download to. We use /home/randompoetry/nltk_data, as that is the first location
# where nltk looks for the data. If we download without specifying the location, it will be downloaded to /usr/local/share, 
# which is not accessible to the user randompoetry.
RUN .venv/bin/python -m nltk.downloader -d ./nltk_data cmudict

# Copy the app code and make the boot.sh script executable
COPY app app
COPY randompoetry randompoetry
COPY data data
COPY config.py setup.py boot.sh ./

# Expose port 5000 (standard Flask port) and run boot.sh (which activates the virtual environment and runs gunicorn)
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
```

# Docker command line

## Build an image

This builds the image from the Dockerfile in the current directory, and tags it as `latest`. To tag with a specific version, use `--tag random-poetry:v1.0`.

```shell
$ docker build --tag random-poetry .
```

## Run an image as a container

- `--publish 8000:5000` - expose the container port 5000 as port 8000 on the host (i.e. on the host, connect to localhost:8000)
- `--env-file=.env' - loads environment variables from the `.env` file and sets them up on the docker container. Useful for injecting secrets and much better than including
them in the Dockerfile
- `--rm` to remove the container when it is done. Otherwise remove with `docker rm ID`
- `--name flask` - name the running container `flask`. This is useful for networking as the other containers will be able to connect to `flask`.
- `--detach` - run the container detached, i.e. run in the background and return to the command prompt.

```shell
$ docker run --name flask --detach --publish 8000:5000 --env-file=.env --rm random-poetry
```

Read more [here](https://vsupalov.com/docker-arg-env-variable-guide/) about environment variables with Docker.

## Stop a container

Run `docker ps` first to see the ID of the container:

```shell
$ docker ps -a
```

Then run `docker stop` to stop the container. It is sufficient to only give the first few unique characters of the ID to stop the container.

```shell
$ docker stop bf7
```

## Remove a container

Details [here](https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes)

```shell
$ docker rm ID
```

## Remove an image

```shell
$ docker rmi ID
```

Use `prune` to remove all dangling images.

```shell
$ docker image prune
``` 