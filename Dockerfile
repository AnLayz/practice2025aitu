FROM python:3.11-slim

# Tesseract и системные зависимости
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "pdf_analyzer.py", "--server.port", "8501", "--server.enableXsrfProtection", "false"]