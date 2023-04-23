FROM docker.io/nersc/pytorch:ngc-20.09-v0

WORKDIR /app

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

