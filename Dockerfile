FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage cache
COPY requirements.txt .

RUN pip install --upgrade pip

# Install dependencies fast
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy code after dependencies
COPY . /app

EXPOSE 7860

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "7860"]