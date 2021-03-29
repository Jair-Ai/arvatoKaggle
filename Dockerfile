# 1. Base image
FROM python:3.7-slim


RUN apt-get update && apt-get install -y libgomp1

RUN pip install pycaret[all]
RUN pip install jupyterlab
#CMD jupyter-lab --allow-root
