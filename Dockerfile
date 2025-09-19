# ใช้ Python เวอร์ชัน 3.10
FROM python:3.10-slim

# ตั้งค่า Working Directory
WORKDIR /app

# Copy ไฟล์ที่จำเป็นทั้งหมด
COPY requirements.txt ./
COPY app.py ./
COPY rf_extract.py ./
COPY cnn_extract.py ./
COPY index.html ./
COPY script.js ./
COPY style.css ./

# ติดตั้ง Library ต่างๆ จาก requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดไฟล์โมเดลขนาดใหญ่จาก Google Drive โดยใช้ gdown
# ***สำคัญ*** gdown จะต้องถูกติดตั้งแล้วในขั้นตอนก่อนหน้า
RUN pip install gdown && \
    gdown --id 1uPsVWj8SjyI71cixpMxQGaZ5F9LnMxH0 -O /app/cough_rf_model.pkl && \
    gdown --id 1BeBGtMNiorzLkiFDaxt2bysji5QNVwT8 -O /app/scaler_rf.pkl

# คำสั่งสำหรับรันแอปพลิเคชันโดยใช้ Gunicorn
# Gunicorn จะเป็นตัวกลางที่ทำให้เว็บเซิร์ฟเวอร์ Flask ของคุณทำงานได้บน Cloud
# Cloud Run จะส่ง Traffic ไปที่พอร์ต 8080 เป็นค่าเริ่มต้น
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
