FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN pip3 install --upgrade pip


RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN useradd -m haythem

RUN chown -R haythem:haythem /home/haythem/

COPY --chown=haythem . /home/haythem/wp_api/

USER haythem

RUN cd /home/haythem/wp_api/ && pip3 install -r requirements.txt

WORKDIR /home/haythem/wp_api