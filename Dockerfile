FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FOUNDRATE_HOME=/data
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY foundrate ./foundrate
COPY docs ./docs
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "foundrate.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
