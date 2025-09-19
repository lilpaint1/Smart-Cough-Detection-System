# ใช้ Python 3.10 slim
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์ requirements แล้วติดตั้ง
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมด
COPY . .

# ตั้ง environment variable สำหรับ Cloud Run
ENV PORT 8080

# เปิด port ให้ Cloud Run
EXPOSE 8080

# รัน Flask app ผ่าน gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
