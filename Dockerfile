FROM python:3.9

WORKDIR ./ColonCancerRepo

ADD . .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install --upgrade pip
RUN pip install numpy==1.24.2
RUN pip install -r requirements.txt

CMD ["python", "./train.py", "./train_config.yaml"]
