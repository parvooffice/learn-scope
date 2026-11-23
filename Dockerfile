FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080

COPY pyproject.toml README.md ./
COPY learnscope ./learnscope
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["python", "-m", "learnscope"]
