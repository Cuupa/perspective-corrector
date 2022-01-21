FROM python:3
MAINTAINER Cuupa
WORKDIR /opt/app/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY __init__.py ./__init__.py

RUN groupadd -r python && useradd --no-log-init -r -g python python
RUN chown python app.py
RUN chown python __init__.py
RUN chmod +x app.py
RUN chmod +x __init__.py
USER python

EXPOSE 5000/tcp
CMD [ "python3", "./app.py" ]
