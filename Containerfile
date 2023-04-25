FROM docker.io/nersc/pytorch:ngc-23.03-v0

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN /sbin/ldconfig

