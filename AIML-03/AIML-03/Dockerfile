FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.1.4 \
    pyarrow==15.0.2 \
    faiss-cpu==1.8.0 \
    redis==5.0.4 \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    "tritonclient[http]==2.45.0" \
    streamlit==1.35.0 \
    pyyaml==6.0.1 \
    requests==2.32.3 \
    httpx==0.27.0
COPY src ./src
COPY configs ./configs
COPY public ./public
COPY deploy_bundle ./deploy_bundle
EXPOSE 8080 8501
