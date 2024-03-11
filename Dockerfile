# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir
ADD src .
RUN python3.11 /moozo_ai/download_models.py

# Add the source code

# Define the command to run your application
CMD python3.11 -u /main.py