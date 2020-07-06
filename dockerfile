FROM python:3

WORKDIR /opt/app/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /opt/app/app.py

RUN groupadd -r python && useradd --no-log-init -r -g python python
RUN chown python app.py
RUN chmod +x app.py
USER python

CMD [ "python3", "app.py" ]
