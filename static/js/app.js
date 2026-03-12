let uploadedImageData = null;
let cameraStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recTimerInterval = null;
let recSeconds = 0;

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');
    document.getElementById(tab + '-tab').classList.add('active');
    document.getElementById(tab + '-content').style.display = 'block';
    if (tab !== 'camera' && cameraStream) stopCamera();
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        uploadedImageData = e.target.result;
        document.getElementById('preview-img').src = uploadedImageData;
        document.getElementById('upload-area').style.display = 'none';
        document.getElementById('upload-preview').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    uploadedImageData = null;
    document.getElementById('file-input').value = '';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('upload-preview').style.display = 'none';
    hidAllResults();
}

function analyzeUploadedImage() {
    if (!uploadedImageData) return;
    analyzeImage(uploadedImageData);
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        const videoEl = document.getElementById('camera-feed');
        videoEl.srcObject = cameraStream;
        videoEl.style.display = 'block';
        document.getElementById('camera-placeholder').style.display = 'none';
        document.getElementById('record-hint').style.display = 'block';

        document.getElementById('start-camera-btn').style.display = 'none';
        document.getElementById('record-btn').style.display = 'block';
        document.getElementById('stop-camera-btn').style.display = 'block';
    } catch (err) {
        alert('Camera access denied or unavailable. Please use the Upload Image tab instead.');
    }
}

function stopCamera() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    clearInterval(recTimerInterval);

    const videoEl = document.getElementById('camera-feed');
    videoEl.style.display = 'none';
    videoEl.srcObject = null;

    document.getElementById('camera-placeholder').style.display = 'block';
    document.getElementById('record-hint').style.display = 'none';
    document.getElementById('rec-badge').style.display = 'none';
    document.getElementById('start-camera-btn').style.display = 'block';
    document.getElementById('record-btn').style.display = 'none';
    document.getElementById('stop-record-btn').style.display = 'none';
    document.getElementById('stop-camera-btn').style.display = 'none';
}

function startRecording() {
    if (!cameraStream) return;
    recordedChunks = [];

    const mimeType = getSupportedMime();
    const options = mimeType ? { mimeType } : {};

    mediaRecorder = new MediaRecorder(cameraStream, options);
    mediaRecorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => sendVideoForAnalysis(mimeType);
    mediaRecorder.start(250);

    recSeconds = 0;
    updateRecTimer();
    recTimerInterval = setInterval(() => {
        recSeconds++;
        updateRecTimer();
    }, 1000);

    document.getElementById('rec-badge').style.display = 'flex';
    document.getElementById('record-btn').style.display = 'none';
    document.getElementById('stop-record-btn').style.display = 'block';
    document.getElementById('stop-camera-btn').style.display = 'none';
    document.getElementById('record-hint').style.display = 'none';
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    clearInterval(recTimerInterval);
    document.getElementById('rec-badge').style.display = 'none';
    document.getElementById('stop-record-btn').style.display = 'none';
}

function updateRecTimer() {
    const m = Math.floor(recSeconds / 60);
    const s = String(recSeconds % 60).padStart(2, '0');
    document.getElementById('rec-timer').textContent = `${m}:${s}`;
}

function getSupportedMime() {
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
    for (const t of types) {
        if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
}

async function sendVideoForAnalysis(mimeType) {
    if (recordedChunks.length === 0) return;

    const type = mimeType || 'video/webm';
    const blob = new Blob(recordedChunks, { type });
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    const file = new File([blob], `swing.${ext}`, { type });

    showLoading('Processing swing video — this may take a moment...');
    hidAllResults();

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/analyze-video', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (result.error) {
            alert('Analysis failed: ' + result.error);
            document.getElementById('results-placeholder').style.display = 'flex';
            return;
        }

        renderSwingResults(result);
    } catch (err) {
        alert('Error connecting to server. Please try again.');
        console.error(err);
        document.getElementById('results-placeholder').style.display = 'flex';
    } finally {
        hideLoading();
        // Restore camera controls
        if (cameraStream) {
            document.getElementById('record-btn').style.display = 'block';
            document.getElementById('stop-camera-btn').style.display = 'block';
            document.getElementById('record-hint').style.display = 'block';
        }
    }
}

async function analyzeImage(imageData) {
    showLoading('Analyzing your swing...');
    hidAllResults();

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();

        if (result.error) {
            alert('Analysis failed: ' + result.error);
            document.getElementById('results-placeholder').style.display = 'flex';
            return;
        }

        document.getElementById('result-img').src = result.image;
        renderFeedback(result.feedback, 'feedback-container');
        document.getElementById('single-results-section').style.display = 'block';
        document.getElementById('single-results-section').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        alert('Error connecting to server. Please try again.');
        document.getElementById('results-placeholder').style.display = 'flex';
    } finally {
        hideLoading();
    }
}

function renderSwingResults(data) {
    const section = document.getElementById('swing-results-section');
    const grid = document.getElementById('phases-grid');
    const meta = document.getElementById('swing-meta');
    const overallBox = document.getElementById('overall-feedback-box');

    meta.textContent = `${data.frame_count} frames · ${data.duration_sec}s recording`;

    // Overall feedback
    if (data.overall && data.overall.length > 0) {
        renderFeedback(data.overall, 'overall-feedback-container');
        overallBox.style.display = 'block';
    }

    // Phase cards
    grid.innerHTML = '';
    (data.phases || []).forEach(phase => {
        const card = document.createElement('div');
        card.className = 'phase-card';

        const goodCount = phase.feedback.filter(f => f.type === 'good').length;
        const warnCount = phase.feedback.filter(f => f.type === 'warning').length;
        const score = goodCount + warnCount > 0
            ? Math.round((goodCount / (goodCount + warnCount)) * 100)
            : 50;

        const scoreColor = score >= 70 ? '#00aa44' : score >= 40 ? '#d29922' : '#cc2200';

        card.innerHTML = `
            <div class="phase-header">
                <span class="phase-name">${phase.phase.toUpperCase()}</span>
                <span class="phase-score" style="color:${scoreColor}">${score}%</span>
            </div>
            <div class="phase-img-wrap">
                <img src="${phase.image}" alt="${phase.phase}">
            </div>
            <div class="phase-feedback">
                ${phase.feedback.map(f => `
                    <div class="feedback-item ${f.type}">
                        <span class="feedback-icon">${f.type === 'good' ? '&#10003;' : '&#9888;'}</span>
                        <span>${f.text}</span>
                    </div>
                `).join('')}
            </div>
        `;
        grid.appendChild(card);
    });

    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });
}

function renderFeedback(feedbackList, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!feedbackList || feedbackList.length === 0) {
        container.innerHTML = '<p style="color:rgba(255,100,100,0.6);font-size:0.9rem;letter-spacing:2px;">No feedback available.</p>';
        return;
    }
    feedbackList.forEach(item => {
        const div = document.createElement('div');
        div.className = 'feedback-item ' + item.type;
        const icon = item.type === 'good' ? '&#10003;' : '&#9888;';
        div.innerHTML = `<span class="feedback-icon">${icon}</span><span>${item.text}</span>`;
        container.appendChild(div);
    });
}

function hidAllResults() {
    document.getElementById('single-results-section').style.display = 'none';
    document.getElementById('swing-results-section').style.display = 'none';
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('overall-feedback-box').style.display = 'none';
}

function showLoading(text) {
    document.getElementById('loading-text').textContent = text || 'Processing...';
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    switchTab('upload');
});
