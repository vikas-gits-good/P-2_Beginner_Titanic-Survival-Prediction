FROM python:3.13-alpine
RUN apk add --no-cache gfortran musl-dev build-base cmake
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools
RUN pip install -r /tmp/requirements.txt
COPY . /CalHsePred
WORKDIR /CalHsePred
CMD python app.py
