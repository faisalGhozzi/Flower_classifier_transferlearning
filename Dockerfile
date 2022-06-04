# The Docker container build file
FROM python:3.9

COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install opencv-python-headless tensorflow==2.7.0 numpy==1.21.3


# What is executed when calling the docker container
ENTRYPOINT [ "python", "main.py" ]