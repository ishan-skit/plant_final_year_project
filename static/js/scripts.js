// ==============================
// PlantCare AI - scripts.js
// ==============================

// Auto-hide flash messages after 5 seconds
setTimeout(() => {
    document.querySelectorAll('.alert').forEach(alert => {
        alert.classList.add('fade');
        setTimeout(() => alert.remove(), 500);
    });
}, 5000);

// Image Upload & Preview (used in predict.html)
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const loadingSpinner = document.querySelector('.loading-spinner');
const predictionForm = document.getElementById('predictionForm');

// Handle file selection
function handleFileSelect() {
    const file = imageInput.files[0];
    if (file) {
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be under 10MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// Remove image
function removeImage() {
    imageInput.value = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
}

// Drag & Drop Support
if (uploadArea && imageInput) {
    uploadArea.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', handleFileSelect);

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            imageInput.files = files;
            handleFileSelect();
        } else {
            alert('Only image files are allowed.');
        }
    });
}

// Prediction form submit
if (predictionForm) {
    predictionForm.addEventListener('submit', (e) => {
        if (!imageInput.files[0]) {
            e.preventDefault();
            alert('Please select an image first.');
            return;
        }
        imagePreview.style.display = 'none';
        loadingSpinner.style.display = 'block';
    });
}

// Smooth scroll for internal anchor links (optional)
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
