# 1. Base image
FROM python:3.7-slim

# 👇 python
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \


# 👇 Tini version
ENV TINI_VERSION="v0.19.0"

# 👇
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
# 👆


# 👇 Not using root to run software
RUN useradd -m -r user && \
    chown user /project

RUN apt-get update && apt-get install -y libgomp1

RUN pip install pycaret[all]

# 👇 Using tini to run software
ENTRYPOINT ["/tini", "--"]