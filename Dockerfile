FROM python:3.8

WORKDIR ./ColonCancerRepo

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "./train.py", "./train_config.yaml"]
