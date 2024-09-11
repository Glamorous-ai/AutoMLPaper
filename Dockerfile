FROM python:3.10-slim

WORKDIR /app

RUN python3 -m pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root
