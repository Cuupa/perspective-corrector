FROM python:3

WORKDIR /opt/app/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /opt/app/app.py

CMD [ "python3", "app.py" ]
