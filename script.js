// --- Constants ---
const LABELS = ['covid', 'healthy', 'symptomatic'];
const SAMPLE_RATE = 44100;
const DURATION = 5; // วินาที

// --- DOM Elements ---
const recordButton = document.getElementById('record-button');
const recordText = document.getElementById('record-text');
const uploadInput = document.getElementById('audio-upload');
const fileNameSpan = document.getElementById('file-name');
const recordSection = document.getElementById('record-section');
const uploadSection = document.getElementById('upload-section');
const resultSection = document.getElementById('result-section');
const predictionLabel = document.getElementById('prediction-label');
const confidenceScore = document.getElementById('confidence-score');
const statusMessage = document.getElementById('status-message');
const probabilityBarContainer = document.getElementById('probability-bar-container');
const resetButton = document.getElementById('reset-button');
const audioBar = document.getElementById('audio-bar');
const messageModal = document.getElementById('message-modal');
const modalTitle = document.getElementById('modal-title');
const modalMessage = document.getElementById('modal-message');
const modalOkButton = document.getElementById('modal-ok-button');

const FLASK_URL = "https://smart-cough-detection-system-412378351230.europe-west1.run.app";


let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyserNode;
let animationFrameId;

// --- Modal helper ---
function showModal(title, message) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    messageModal.classList.remove('hidden');
}

// --- Record button ---
recordButton.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];

        // Waveform visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);

        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

        mediaRecorder.onstop = async () => {
            cancelAnimationFrame(animationFrameId);
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const wavBlob = await convertToWav(audioBlob);
            processAudio(wavBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        recordText.textContent = "กำลังบันทึก...";
        recordButton.classList.remove('bg-teal-500', 'hover:bg-teal-600');
        recordButton.classList.add('bg-red-500', 'hover:bg-red-600');
        uploadInput.disabled = true;
        visualizeAudio();

        setTimeout(() => {
            if (mediaRecorder.state === 'recording') mediaRecorder.stop();
        }, DURATION * 1000);

    } catch (err) {
        showModal("ข้อผิดพลาด", `ไม่สามารถเข้าถึงไมโครโฟนได้: ${err.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
    recordText.textContent = "เริ่มบันทึก";
    recordButton.classList.remove('bg-red-500', 'hover:bg-red-600');
    recordButton.classList.add('bg-teal-500', 'hover:bg-teal-600');
    uploadInput.disabled = false;
}

function visualizeAudio() {
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    const updateBar = () => {
        analyserNode.getByteFrequencyData(dataArray);
        let sum = dataArray.reduce((a,b)=>a+b,0);
        let avg = sum / dataArray.length;
        audioBar.style.width = `${(avg/255)*100}%`;
        animationFrameId = requestAnimationFrame(updateBar);
    };
    updateBar();
}

// --- File upload ---
uploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        fileNameSpan.textContent = `ไฟล์ที่เลือก: ${file.name}`;
        convertFileToWav(file).then(wavBlob => processAudio(wavBlob));
    }
});

// --- Convert audio/webm (from mic) or other formats to WAV ---
async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const numChannels = 1;
    const length = audioBuffer.length;
    const sampleRate = SAMPLE_RATE;
    const wavBuffer = audioBufferToWav(audioBuffer, numChannels, sampleRate);

    return new Blob([wavBuffer], { type: 'audio/wav' });
}

// --- Convert uploaded file to WAV if not wav ---
async function convertFileToWav(file) {
    if (file.type === "audio/wav") return file;
    return convertToWav(file);
}

// --- Helper: AudioBuffer → WAV array buffer ---
function audioBufferToWav(buffer, numChannels, sampleRate) {
    const channels = [];
    for (let i=0; i<numChannels; i++) {
        channels.push(buffer.getChannelData(i % buffer.numberOfChannels));
    }
    const interleaved = interleave(channels);
    const bufferLength = 44 + interleaved.length * 2;
    const wav = new ArrayBuffer(bufferLength);
    const view = new DataView(wav);

    /* RIFF chunk */
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + interleaved.length * 2, true);
    writeString(view, 8, 'WAVE');

    /* fmt subchunk */
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true); // bits per sample

    /* data subchunk */
    writeString(view, 36, 'data');
    view.setUint32(40, interleaved.length * 2, true);

    // Write PCM samples
    let offset = 44;
    for (let i=0; i<interleaved.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, interleaved[i]));
        view.setInt16(offset, s < 0 ? s*0x8000 : s*0x7FFF, true);
    }
    return wav;
}

function interleave(channels) {
    const length = channels[0].length;
    const result = new Float32Array(length * channels.length);
    for (let i=0; i<length; i++) {
        for (let j=0; j<channels.length; j++) {
            result[i*channels.length + j] = channels[j][i];
        }
    }
    return result;
}

function writeString(view, offset, string) {
    for (let i=0; i<string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// --- Send to Flask ---
async function processAudio(audioBlob) {
    recordSection.classList.add('hidden');
    uploadSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
    statusMessage.textContent = "🤖 กำลังประมวลผลเสียงและทำนาย...";

    try {
        const formData = new FormData();
        formData.append("file", audioBlob, "cough.wav");

        const response = await fetch(FLASK_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const result = await response.json();
        if (result.error) throw new Error(result.error);

        displayResult(result);

    } catch (error) {
        console.error("Prediction Error:", error);
        statusMessage.textContent = "❌ เกิดข้อผิดพลาดในการประมวลผล";
        predictionLabel.textContent = "ไม่สามารถทำนายผลได้";
        predictionLabel.style.color = "red";
        resetButton.classList.remove('hidden');
    }
}

function displayResult(data) {
    statusMessage.textContent = "✅ เสร็จสิ้นการวิเคราะห์";
    
    const classifiedLabel = data.classification;
    const probabilities = data.probabilities;

    const highestScore = probabilities.reduce((max, prob) => Math.max(max, prob.score), 0);
    
    predictionLabel.textContent = classifiedLabel.toUpperCase();
    confidenceScore.textContent = `(ความมั่นใจ: ${(highestScore*100).toFixed(2)}%)`;
    predictionLabel.style.color = classifiedLabel === 'healthy' ? "#4ade80" : "#ef4444";

    probabilityBarContainer.innerHTML = '';
    probabilities.forEach(prob => {
        const barWrapper = document.createElement('div');
        barWrapper.className = "flex items-center gap-2 text-sm";
        
        const label = document.createElement('span');
        label.className = "w-24 text-right";
        label.textContent = prob.label;
        
        const progressBar = document.createElement('div');
        progressBar.className = "flex-1 bg-gray-600 rounded-full h-2.5 overflow-hidden";
        
        const progress = document.createElement('div');
        progress.className = "bg-teal-400 h-full rounded-full transition-all duration-500 ease-out";
        progress.style.width = `${prob.score*100}%`;
        progressBar.appendChild(progress);
        
        const score = document.createElement('span');
        score.className = "w-12 text-left";
        score.textContent = `${(prob.score*100).toFixed(1)}%`;
        
        barWrapper.appendChild(label);
        barWrapper.appendChild(progressBar);
        barWrapper.appendChild(score);
        probabilityBarContainer.appendChild(barWrapper);
    });
    
    resetButton.classList.remove('hidden');
}

// --- Reset ---
resetButton.addEventListener('click', () => {
    recordSection.classList.remove('hidden');
    uploadSection.classList.remove('hidden');
    resultSection.classList.add('hidden');
    fileNameSpan.textContent = '';
    audioBar.style.width = '0%';
    recordButton.disabled = false;
});

// --- Close modal ---
modalOkButton.addEventListener('click', () => messageModal.classList.add('hidden'));

