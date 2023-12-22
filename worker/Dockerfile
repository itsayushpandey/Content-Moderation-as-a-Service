# FROM xserrat/facebook-demucs:latest
# FROM rajeevrmenon97/demucs-cpu:latest
FROM python:3.9-slim

USER root
ENV TORCH_HOME=/data/models
RUN apt update -y # buildkit
RUN apt install -y ffmpeg git # buildkit
RUN python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu # buildkit

RUN python3 -m pip install diffq demucs # buildkit

RUN pip3 install --upgrade pip

RUN pip3 install redis minio jsonpickle

# VOLUME [/data/input]
# VOLUME [/data/output]
# VOLUME [/data/models]

# RUN python3 -m pip install -U demucs

WORKDIR /srv
COPY    worker.py /srv
CMD [ "python3", "worker.py"]
