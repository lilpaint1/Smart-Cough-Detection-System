// ============================================================
// CoughAI — script.js (Premium UI Rebuild)
// Backend API, audio logic, and data flow are 100% unchanged.
// Only UI interactions, states, and animations have been updated.
// ============================================================

// --- Constants (unchanged) ---
const LABELS = ['covid', 'healthy', 'symptomatic'];
const SAMPLE_RATE = 44100;
const DURATION = 5;

// --- DOM Elements ---
const recordButton         = document.getElementById('record-button');
const recordText           = document.getElementById('record-text');
const uploadInput          = document.getElementById('audio-upload');
const fileNameSpan         = document.getElementById('file-name');
const recordSection        = document.getElementById('record-section');
const uploadSection        = document.getElementById('upload-section');
const resultSection        = document.getElementById('result-section');
const resultLoading        = document.getElementById('result-loading');
const resultCard           = document.getElementById('result-card');
const predictionLabel      = document.getElementById('prediction-label');
const confidenceScore      = document.getElementById('confidence-score');
const statusMessage        = document.getElementById('status-message');
const probabilityBarContainer = document.getElementById('probability-bar-container');
const resetButton          = document.getElementById('reset-button');
const audioBar             = document.getElementById('audio-bar');
const waveformLabel        = document.getElementById('waveform-label');
const messageModal         = document.getElementById('message-modal');
const modalTitle           = document.getElementById('modal-title');
const modalMessage         = document.getElementById('modal-message');
const modalOkButton        = document.getElementById('modal-ok-button');
const resultBadge          = document.getElementById('result-badge');
const badgeIcon            = document.getElementById('badge-icon');
const dropZone             = document.getElementById('drop-zone');

// --- API URL (unchanged) ---
const FLASK_URL = "https://coughai-985046969554.asia-southeast3.run.app/predict";

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyserNode;
let animationFrameId;

// ─── Modal helper ────────────────────────────────────────────
function showModal(title, message) {
    modalTitle.textContent   = title;
    modalMessage.textContent = message;
    messageModal.classList.remove('hidden');
}

// ─── Record button ───────────────────────────────────────────
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
            const wavBlob   = await convertToWav(audioBlob);
            processAudio(wavBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();

        // ── UI: recording state ──
        recordText.textContent = 'กำลังบันทึก...';
        recordButton.classList.add('recording');
        if (waveformLabel) waveformLabel.textContent = 'กำลังรับเสียง...';
        uploadInput.disabled = true;

        visualizeAudio();

        // Auto-stop after DURATION seconds
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') mediaRecorder.stop();
        }, DURATION * 1000);

    } catch (err) {
        showModal('ข้อผิดพลาด', `ไม่สามารถเข้าถึงไมโครโฟนได้: ${err.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();

    // ── UI: idle state ──
    recordText.textContent = 'เริ่มบันทึก';
    recordButton.classList.remove('recording');
    if (waveformLabel) waveformLabel.textContent = 'รอสัญญาณเสียง...';
    audioBar.style.width = '0%';
    uploadInput.disabled = false;
}

function visualizeAudio() {
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    const update = () => {
        analyserNode.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        audioBar.style.width = `${(avg / 255) * 100}%`;
        animationFrameId = requestAnimationFrame(update);
    };
    update();
}

// ─── File Upload ─────────────────────────────────────────────
uploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        fileNameSpan.textContent = file.name;
        convertFileToWav(file).then(wavBlob => processAudio(wavBlob));
    }
});

// Drag-and-drop enhancement
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            fileNameSpan.textContent = file.name;
            convertFileToWav(file).then(wavBlob => processAudio(wavBlob));
        } else {
            showModal('ไฟล์ไม่ถูกต้อง', 'กรุณาเลือกไฟล์เสียงเท่านั้น (.wav, .mp3, .webm)');
        }
    });
}

// ─── Audio Conversion (unchanged logic) ──────────────────────
async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx    = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const wavBuffer   = audioBufferToWav(audioBuffer, 1, SAMPLE_RATE);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

async function convertFileToWav(file) {
    if (file.type === 'audio/wav') return file;
    return convertToWav(file);
}

function audioBufferToWav(buffer, numChannels, sampleRate) {
    const channels = [];
    for (let i = 0; i < numChannels; i++) {
        channels.push(buffer.getChannelData(i % buffer.numberOfChannels));
    }
    const interleaved   = interleave(channels);
    const bufferLength  = 44 + interleaved.length * 2;
    const wav  = new ArrayBuffer(bufferLength);
    const view = new DataView(wav);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + interleaved.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, interleaved.length * 2, true);

    let offset = 44;
    for (let i = 0; i < interleaved.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, interleaved[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return wav;
}

function interleave(channels) {
    const length = channels[0].length;
    const result = new Float32Array(length * channels.length);
    for (let i = 0; i < length; i++) {
        for (let j = 0; j < channels.length; j++) {
            result[i * channels.length + j] = channels[j][i];
        }
    }
    return result;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ─── Send to Flask (unchanged logic) ─────────────────────────
async function processAudio(audioBlob) {
    // Show result section with loading state
    recordSection.classList.add('hidden');
    uploadSection.classList.add('hidden');

    const divider = document.querySelector('.divider-or');
    if (divider) divider.classList.add('hidden');

    resultSection.classList.remove('hidden');
    resultLoading.classList.remove('hidden');
    resultCard.classList.add('hidden');

    statusMessage.textContent = '🤖 กำลังประมวลผลเสียงและทำนาย...';

    try {
        const formData = new FormData();
        formData.append('file', audioBlob, 'cough.wav');

        const response = await fetch(FLASK_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const result = await response.json();
        if (result.error) throw new Error(result.error);

        // Small delay so loading animation is perceptible
        await new Promise(r => setTimeout(r, 400));
        displayResult(result);

    } catch (error) {
        console.error('Prediction Error:', error);
        displayError(error.message);
    }
}

// ─── Display Result ───────────────────────────────────────────
function displayResult(data) {
    resultLoading.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const classifiedLabel = data.classification.toLowerCase();
    const probabilities   = data.probabilities;
    const highestScore    = probabilities.reduce((max, p) => Math.max(max, p.score), 0);

    // Label text + color class
    const labelMap = {
        healthy:     { text: 'Healthy',     cls: 'label-healthy',     badge: 'badge-healthy'     },
        covid:       { text: 'COVID-19',    cls: 'label-covid',       badge: 'badge-covid'       },
        symptomatic: { text: 'Symptomatic', cls: 'label-symptomatic', badge: 'badge-symptomatic' },
    };
    const match = labelMap[classifiedLabel] || { text: classifiedLabel.toUpperCase(), cls: '', badge: '' };

    predictionLabel.textContent = match.text;
    predictionLabel.className   = `result-label ${match.cls}`;

    confidenceScore.textContent = `ความมั่นใจ ${(highestScore * 100).toFixed(1)}%`;

    // Badge icon — use ✓ for healthy, ✕ for others
    const isHealthy = classifiedLabel === 'healthy';
    resultBadge.className = `result-badge ${match.badge}`;
    badgeIcon.innerHTML   = isHealthy
        ? '<polyline points="20 6 9 17 4 12"/>'
        : classifiedLabel === 'covid'
            ? '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>'
            : '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>';

    // Probability bars
    probabilityBarContainer.innerHTML = '';

    const colorMap = {
        healthy:     'healthy',
        covid:       'covid',
        symptomatic: 'symptomatic',
    };

    probabilities.forEach(prob => {
        const row = document.createElement('div');
        row.className = 'prob-row';

        const label = document.createElement('span');
        label.className   = 'prob-label';
        label.textContent = prob.label;

        const track = document.createElement('div');
        track.className = 'prob-track';

        const fill = document.createElement('div');
        const colorCls = colorMap[prob.label.toLowerCase()] || 'default';
        fill.className = `prob-fill ${colorCls}`;
        fill.style.width = '0%';
        track.appendChild(fill);

        const score = document.createElement('span');
        score.className   = 'prob-score';
        score.textContent = `${(prob.score * 100).toFixed(1)}%`;

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(score);
        probabilityBarContainer.appendChild(row);

        // Animate bar after paint
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                fill.style.width = `${prob.score * 100}%`;
            });
        });
    });
}

// ─── Display Error ────────────────────────────────────────────
function displayError(errorMsg) {
    resultLoading.classList.add('hidden');
    resultCard.classList.remove('hidden');

    predictionLabel.textContent = 'วิเคราะห์ไม่สำเร็จ';
    predictionLabel.className   = 'result-label label-error';
    confidenceScore.textContent = '';
    probabilityBarContainer.innerHTML = '';
    resultBadge.className = 'result-badge badge-error';
    badgeIcon.innerHTML   = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>';
}

// ─── Reset ────────────────────────────────────────────────────
resetButton.addEventListener('click', () => {
    resultSection.classList.add('hidden');
    resultCard.classList.add('hidden');
    resultLoading.classList.add('hidden');

    recordSection.classList.remove('hidden');
    uploadSection.classList.remove('hidden');

    const divider = document.querySelector('.divider-or');
    if (divider) divider.classList.remove('hidden');

    fileNameSpan.textContent = '';
    audioBar.style.width     = '0%';
    if (waveformLabel) waveformLabel.textContent = 'รอสัญญาณเสียง...';
    uploadInput.value         = '';
    recordButton.disabled     = false;
});

// ─── Close Modal ──────────────────────────────────────────────
modalOkButton.addEventListener('click', () => messageModal.classList.add('hidden'));
