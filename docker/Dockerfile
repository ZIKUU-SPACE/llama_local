FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

RUN apt-get update && apt-get upgrade -y
RUN apt-get dist-upgrade -y && apt-get autoremove -y
RUN apt-get install -y build-essential wget
RUN apt-get install -y python3 python3-venv

WORKDIR /app

RUN wget -P models https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf

COPY ./requirements.txt /app
COPY ./webui.py /app

RUN python3 -m venv /venv
RUN . /venv/bin/activate && python -m pip install --upgrade pip
RUN . /venv/bin/activate && python -m pip install -r requirements.txt


COPY ./docker/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]