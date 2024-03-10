# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir

WORKDIR /src/models
COPY download_models.py /src/download_models.py
RUN python3.11 /src/download_models.py

WORKDIR /src

# Add the source code
ADD src .

# Define the command to run your application
CMD ["python3.11", "-u", "/src/main.py"]
