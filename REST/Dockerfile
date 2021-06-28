FROM ubuntu:xenial-20210429

ENV LISTEN_PORT=7000
EXPOSE 7000

RUN apt-get update -y
RUN apt-get install -y python3.5 python3-pip build-essential

COPY . /home
WORKDIR /home

RUN python3 -m pip install --upgrade "pip < 21.0"
RUN python3 -m pip install -r lib.txt

CMD ["python3", "-u", "main.py"]
