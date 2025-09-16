import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

def extract_features_rf(file_path_or_array, sr=44100,duration=10):
    """
    Extracts traditional audio features for RandomForestClassifier.
    
    Args:
        file_path_or_array (str or np.ndarray): Path to the audio file or the audio signal as a numpy array.
        sr (int): Sampling rate, required if passing a numpy array.
        duration (int): Duration of the audio to load in seconds.
        
    Returns:
        np.array: A horizontal stack of audio features, or None on error.
    """
    try:
        if isinstance(file_path_or_array, np.ndarray):
            y = file_path_or_array
            if sr is None:
                raise ValueError("Sampling rate 'sr' must be provided when passing a numpy array.")
        else:
            y, sr = librosa.load(file_path_or_array, duration=duration, sr=sr)

        # Ensure the audio signal is the correct duration
        y = librosa.util.fix_length(y, size=sr * duration)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Stack features into a single array
        features = np.hstack([
            mfccs, 
            np.array([spec_centroid]), 
            np.array([zcr]), 
            chroma, 
            np.array([rolloff]) 
        ])
        
        return features

    except Exception as e:
        print(f"❌ Error extracting features from {file_path_or_array}: {e}")
        return None

def process_and_save_data_rf(base_path, classes, output_npz="features_rf.npz", output_scaler="scaler_rf.pkl"):
    """
    Processes audio files, extracts features for RF, and saves them.
    """
    X, y = [], []
    print("🔄 เริ่มสกัดฟีเจอร์สำหรับ RandomForest...")

    for idx, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)
        print(f"📁 กำลังประมวลผลคลาส: {cls}")
        for file in tqdm(os.listdir(folder), desc=f"🔍 {cls}", unit="file"):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                features = extract_features_rf(path)
                if features is not None:
                    X.append(features)
                    y.append(idx)

    X = np.array(X)
    y = np.array(y)

    print("📏 กำลัง Normalize ข้อมูล...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, output_scaler)
    
    np.savez(output_npz, X=X_scaled, y=y)
    print(f"💾 บันทึกข้อมูลที่ Normalize แล้วไปที่ {output_npz}")
    print(f"💾 บันทึก scaler ไปที่ {output_scaler}")

if __name__ == "__main__":
    base_path = r"C:\Users\Acer\Downloads\Cough Detection\public_dataset\sorted_audio"
    classes = ["covid", "healthy", "symptomatic"]
    process_and_save_data_rf(base_path, classes)