FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y ghostscript
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]