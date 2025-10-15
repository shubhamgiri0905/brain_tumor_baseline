# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Working directory inside container
WORKDIR /app

# 3️⃣ Copy project files
COPY . /app

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Expose port
EXPOSE 8000

# 6️⃣ Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
