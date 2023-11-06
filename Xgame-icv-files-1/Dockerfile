FROM python:3.9-bullseye

RUN mkdir /code
RUN pip install pulp
RUN pip install numpy

COPY *.py /code/
COPY *.whl /code/
COPY requirements.txt /code/
COPY X-Y_map.json /code/

WORKDIR /code

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install *.whl

ENTRYPOINT ["python", "-u", "demo.py"]