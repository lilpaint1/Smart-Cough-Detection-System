# ใช้ Python 3.10 slim
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์ requirements แล้วติดตั้ง
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดลและ scaler จาก Google Drive ตั้งแต่ตอน Build time
# ติดตั้ง gdown ก่อนเพื่อใช้ดาวน์โหลดไฟล์
RUN pip install gdown && \
    gdown --id 1uPsVWj8SjyI71cixpMxQGaZ5F9LnMxH0 -O /app/cough_rf_model.pkl && \
    gdown --id 1BVfM7bgBw0XX4fn5Vh_gcitQ91IwRVFv -O /app/scaler_rf.pkl

# คัดลอกเฉพาะไฟล์โค้ดที่จำเป็น
COPY app.py ./
COPY rf_extract.py ./
COPY cnn_extract.py ./
COPY index.html ./
COPY script.js ./
COPY style.css ./

# ตั้งค่าให้ Gunicorn รัน Flask app ที่ชื่อ app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
