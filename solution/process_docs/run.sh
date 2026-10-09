#!/bin/bash

export LOGGER_LEVEL=INFO
export S3_ENDPOINT=http://localhost:9000
export S3_REGION=ru-central1
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export GLINER2_MODEL_NAME=fastino/gliner2-multi-v1
export GLINER2_MODEL_INP_TOKENS=512
export ENCODER_MODEL_NAME=BAAI/bge-base-en-v1.5
export QDRANT_URL=http://localhost:6333

python app.py