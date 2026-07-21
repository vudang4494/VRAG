FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fonts-liberation \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY dashboard/requirements.txt /tmp/reqs.txt
RUN pip install --no-cache-dir -r /tmp/reqs.txt

COPY dashboard/ /app/

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "dashboard.py"]
