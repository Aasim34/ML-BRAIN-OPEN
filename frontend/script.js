// Brain Tumor Classification - Frontend JavaScript

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const imageInfo = document.getElementById('imageInfo');
const clearBtn = document.getElementById('clearBtn');
const actionSection = document.getElementById('actionSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const modelStatus = document.getElementById('modelStatus');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');
const downloadBtn = document.getElementById('downloadBtn');

// State
let currentFile = null;
let currentResults = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Buttons
    clearBtn.addEventListener('click', clearImage);
    analyzeBtn.addEventListener('click', analyzeImage);
    newAnalysisBtn.addEventListener('click', resetApp);
    retryBtn.addEventListener('click', resetApp);
    downloadBtn.addEventListener('click', downloadReport);
}

// Check Model Status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const indicator = modelStatus.querySelector('.status-indicator');
        const statusText = modelStatus.querySelector('.status-text');
        
        if (data.model_loaded) {
            indicator.classList.add('active');
            statusText.textContent = 'Model Ready';
        } else {
            indicator.classList.add('error');
            statusText.textContent = 'Model Not Loaded';
        }
    } catch (error) {
        const indicator = modelStatus.querySelector('.status-indicator');
        const statusText = modelStatus.querySelector('.status-text');
        indicator.classList.add('error');
        statusText.textContent = 'API Offline';
        console.error('Error checking model status:', error);
    }
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Please drop an image file');
    }
}

function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload JPG, PNG, BMP, or TIFF images.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds 16MB limit.');
        return;
    }

    currentFile = file;

    // Read and display image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.style.display = 'block';
        actionSection.style.display = 'block';
        
        // Display file info
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        imageInfo.innerHTML = `
            <strong>File:</strong> ${file.name}<br>
            <strong>Size:</strong> ${fileSizeMB} MB<br>
            <strong>Type:</strong> ${file.type}
        `;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.style.display = 'none';
    actionSection.style.display = 'none';
    previewImage.src = '';
    imageInfo.innerHTML = '';
}

// Analyze Image
async function analyzeImage() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }

    // Hide previous sections
    actionSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Show loading
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentResults = data.data;
            displayResults(data.data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the API. Please ensure the backend is running.');
    } finally {
        loadingSection.style.display = 'none';
    }
}

// Display Results
function displayResults(results) {
    const { predicted_class, confidence, all_predictions } = results;

    // Set result badge
    const resultBadge = document.getElementById('resultBadge');
    resultBadge.textContent = predicted_class.toUpperCase();
    resultBadge.className = `result-badge ${predicted_class}`;

    // Primary result
    const primaryResult = document.getElementById('primaryResult');
    primaryResult.innerHTML = `
        <div class="diagnosis">${formatClassName(predicted_class)}</div>
        <div class="confidence-text">${(confidence * 100).toFixed(2)}% Confidence</div>
    `;

    // Confidence meter
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceLabel = document.getElementById('confidenceLabel');
    
    setTimeout(() => {
        confidenceFill.style.width = `${confidence * 100}%`;
        confidenceFill.textContent = `${(confidence * 100).toFixed(1)}%`;
    }, 100);

    const confidenceLevel = getConfidenceLevel(confidence);
    confidenceLabel.textContent = `Confidence Level: ${confidenceLevel}`;

    // All predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';

    all_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;
        
        item.innerHTML = `
            <span class="prediction-class">${formatClassName(pred.class)}</span>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: 0%"></div>
            </div>
            <span class="prediction-percentage">${pred.percentage}</span>
        `;
        
        predictionsList.appendChild(item);

        // Animate bar
        setTimeout(() => {
            const fill = item.querySelector('.prediction-fill');
            fill.style.width = `${pred.probability * 100}%`;
        }, 100 + (index * 100));
    });

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Utility Functions
function formatClassName(className) {
    const names = {
        'glioma': 'Glioma',
        'meningioma': 'Meningioma',
        'pituitary': 'Pituitary Tumor',
        'notumor': 'No Tumor Detected'
    };
    return names[className] || className;
}

function getConfidenceLevel(confidence) {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.75) return 'High';
    if (confidence >= 0.6) return 'Moderate';
    return 'Low';
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.classList.add('fade-in');
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetApp() {
    clearImage();
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'none';
    currentResults = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Download Report
function downloadReport() {
    if (!currentResults) return;

    const report = generateReport();
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `brain_tumor_report_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function generateReport() {
    const { predicted_class, confidence, all_predictions, metadata } = currentResults;
    const date = new Date().toLocaleString();

    let report = `
================================
BRAIN TUMOR CLASSIFICATION REPORT
================================

Report Generated: ${date}

DIAGNOSIS
---------
Predicted Class: ${formatClassName(predicted_class)}
Confidence: ${(confidence * 100).toFixed(2)}%
Confidence Level: ${getConfidenceLevel(confidence)}

ALL PREDICTIONS
---------------
${all_predictions.map(pred => 
    `${formatClassName(pred.class).padEnd(25)} ${pred.percentage}`
).join('\n')}

FILE INFORMATION
----------------
Original File: ${metadata?.original_filename || 'N/A'}
Image Size: ${metadata?.image_size?.join(' x ') || 'N/A'}
Timestamp: ${metadata?.timestamp || 'N/A'}

IMPORTANT NOTICE
----------------
This classification is provided by an AI model for research and 
educational purposes only. It should NOT be used as a sole basis 
for medical diagnosis. Always consult qualified medical professionals 
for proper diagnosis and treatment.

================================
`;

    return report.trim();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + O to open file
    if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }
    
    // Escape to clear/reset
    if (e.key === 'Escape') {
        resetApp();
    }
});

// Prevent accidental page refresh
window.addEventListener('beforeunload', (e) => {
    if (currentFile && !currentResults) {
        e.preventDefault();
        e.returnValue = '';
    }
});

console.log('🧠 Brain Tumor Classification - Frontend Loaded');
console.log('API Base URL:', API_BASE_URL);
