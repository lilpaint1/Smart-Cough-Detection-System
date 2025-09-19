# ใช้ Python 3.10 slim
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์ requirements.txt แล้วติดตั้ง packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดลและ scaler จาก Google Drive ตั้งแต่ตอน Build time
RUN pip install gdown && \
    gdown --id 1uPsVWj8SjyI71cixpMxQGaZ5F9LnMxH0 -O /app/cough_rf_model.pkl && \
    gdown --id 1BVfM7bgBw0XX4fn5Vh_gcitQ91IwRVFv -O /app/scaler_rf.pkl

# คัดลอกเฉพาะไฟล์โค้ดที่จำเป็น
COPY app.py ./
COPY rf_extract.py ./

# Gunicorn ใช้ Port 8080 เป็นค่าเริ่มต้นของ Cloud Run อยู่แล้ว
# ดังนั้นไม่จำเป็นต้องตั้ง ENV PORT และ EXPOSE
# ตั้งค่า Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
