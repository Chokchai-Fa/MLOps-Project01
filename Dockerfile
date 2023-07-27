# FROM jupyter/scipy-notebook
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

RUN pip install joblib
RUN pip install pandas
RUN pip install -U scikit-learn scipy matplotlib

USER root
RUN apt-get update && apt-get install -y jq

RUN mkdir model raw_data processed_data results


ENV RAW_DATA_DIR=/raw_data
ENV PROCESSED_DATA_DIR=/processed_data
ENV MODEL_DIR=/model
ENV RESULTS_DIR=/results
ENV RAW_DATA_FILE=heart.csv


COPY heart.csv ./raw_data/heart.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py