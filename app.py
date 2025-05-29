# Flask Plant Disease Detection App - Production Deployment Ready
import cv2
import base64
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
import logging
from io import BytesIO
from PIL import Image
import threading
import time
import signal
import sys
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Enhanced CPU-only mode and memory optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit CPU threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Global variables for lazy loading
TENSORFLOW_AVAILABLE = False
tf = None
model = None
label_dict = {}
MODEL_LOADED = False
MODEL_LOADING = False
model_loading_lock = threading.RLock()
app_ready = threading.Event()

def lazy_import_tensorflow():
    """Import TensorFlow only when needed with memory optimization"""
    global tf, TENSORFLOW_AVAILABLE
    if tf is None:
        try:
            import tensorflow as tf_import
            tf = tf_import
            
            # Enhanced CPU configuration
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            # Memory optimization
            import gc
            gc.collect()
            
            TENSORFLOW_AVAILABLE = True
            logger.info("TensorFlow imported successfully with CPU optimization")
        except Exception as e:
            logger.error(f"TensorFlow import failed: {e}")
            TENSORFLOW_AVAILABLE = False
    return TENSORFLOW_AVAILABLE

# Optional imports with fallbacks
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    logger.warning("python-dotenv not available")

try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True if os.getenv("GEMINI_API_KEY") else False
except:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini AI not available")

# App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB

# Setup upload directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Removed gif to reduce processing
MODEL_PATH = os.path.join('model', 'model.h5')
LABELS_PATH = os.path.join('model', 'labels.json')
TREATMENTS_PATH = 'plant_treatments.csv'

# Global variables
treatments_df = pd.DataFrame()

def check_files():
    """Check if required files exist"""
    files_status = {
        'model': os.path.exists(MODEL_PATH),
        'labels': os.path.exists(LABELS_PATH),
        'treatments': os.path.exists(TREATMENTS_PATH)
    }
    logger.info(f"File status: {files_status}")
    return files_status

def load_model_safe():
    """Load model with enhanced safety and timeout protection"""
    global model, label_dict, MODEL_LOADED, MODEL_LOADING
    
    with model_loading_lock:
        if MODEL_LOADED or MODEL_LOADING:
            return MODEL_LOADED
            
        MODEL_LOADING = True
        logger.info("Starting safe model loading...")
        
        try:
            # Import TensorFlow first
            if not lazy_import_tensorflow():
                logger.error("TensorFlow not available")
                MODEL_LOADING = False
                return False
            
            # Check files exist
            file_status = check_files()
            if not file_status['model'] or not file_status['labels']:
                logger.error("Required model files not found")
                MODEL_LOADING = False
                return False
            
            # Load labels first (faster)
            with open(LABELS_PATH, 'r') as f:
                label_map = json.load(f)
            label_dict = {v: k for k, v in label_map.items()}
            logger.info(f"Labels loaded: {len(label_dict)} classes")
            
            # Load model with memory optimization
            with tf.device('/CPU:0'):
                # Load with reduced memory footprint
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    compile=False,
                    options=tf.saved_model.LoadOptions()
                )
                
                # Compile with minimal settings
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    run_eagerly=False
                )
                
                # Quick test with small input
                dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
                _ = model.predict(dummy_input, verbose=0)
                
                # Clean up memory
                del dummy_input
                gc.collect()
                
            MODEL_LOADED = True
            MODEL_LOADING = False
            logger.info("✅ Model loaded successfully with memory optimization")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            model = None
            label_dict = {}
            MODEL_LOADED = False
            MODEL_LOADING = False
            gc.collect()  # Clean up on error
            return False

def load_treatments_safe():
    """Safely load treatment data"""
    global treatments_df
    try:
        if os.path.exists(TREATMENTS_PATH):
            treatments_df = pd.read_csv(TREATMENTS_PATH)
            logger.info(f"Treatment data loaded: {len(treatments_df)} treatments")
        else:
            logger.warning(f"Treatment file not found: {TREATMENTS_PATH}")
            treatments_df = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading treatments: {e}")
        treatments_df = pd.DataFrame()

def init_db():
    """Initialize database with connection pooling"""
    try:
        conn = sqlite3.connect('plant_app.db', timeout=10.0)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize components
logger.info("Initializing application...")
init_db()
load_treatments_safe()

# Set app ready immediately for health checks
app_ready.set()

# Start background model loading (non-blocking)
def background_model_loader():
    """Background thread for model loading"""
    time.sleep(2)  # Small delay to let app start
    load_model_safe()

model_thread = threading.Thread(target=background_model_loader, daemon=True)
model_thread.start()

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_safe(img_path):
    """Memory-efficient image preprocessing"""
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def predict_disease_safe(img_path):
    """Safe disease prediction with proper error handling"""
    if not MODEL_LOADED or model is None:
        if MODEL_LOADING:
            return "Model is loading, please wait...", 0.0
        else:
            return "Model not available", 0.0
    
    try:
        logger.info(f"Starting prediction for: {img_path}")
        img_array = preprocess_image_safe(img_path)
        
        with tf.device('/CPU:0'):
            predictions = model.predict(img_array, verbose=0)
        
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        class_name = label_dict.get(class_index, f"Class_{class_index}")
        
        # Clean up memory
        del img_array, predictions
        gc.collect()
        
        logger.info(f"Prediction: {class_name} ({confidence:.2%})")
        return class_name, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Prediction Error", 0.0

def get_treatment_info(disease_name, confidence=0.0):
    """Get treatment information"""
    default_treatment = {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Remove affected plant parts, improve air circulation, ensure proper drainage. Consult an agricultural expert for specific treatment recommendations.',
        'prevention': 'Maintain proper plant spacing, ensure good drainage, monitor plants regularly, and provide appropriate nutrition.',
        'organic_treatment': 'Consider using neem oil, copper-based fungicides, or organic compost to improve plant health.',
        'chemical_treatment': 'Consult with agricultural extension services for appropriate chemical treatment options.',
        'severity': 'Monitor closely and take appropriate action based on symptom progression.',
        'source': 'General Guidelines'
    }
    
    if not treatments_df.empty:
        treatment_row = treatments_df[treatments_df['disease'].str.contains(disease_name, case=False, na=False)]
        if not treatment_row.empty:
            row = treatment_row.iloc[0]
            return {
                'disease': row.get('disease', disease_name),
                'treatment': row.get('treatment', default_treatment['treatment']),
                'prevention': row.get('prevention', default_treatment['prevention']),
                'organic_treatment': row.get('organic_treatment', default_treatment['organic_treatment']),
                'chemical_treatment': row.get('chemical_treatment', default_treatment['chemical_treatment']),
                'severity': row.get('severity', default_treatment['severity']),
                'source': 'Database'
            }
    
    return default_treatment

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/health')
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        "status": "healthy",
        "app_ready": app_ready.is_set(),
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "files": check_files(),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/ready')
def readiness_check():
    """Kubernetes readiness probe"""
    if app_ready.is_set():
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "not ready"}), 503

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not all([username, email, password]):
            flash("All fields are required!")
            return redirect(request.url)

        try:
            hashed_pw = generate_password_hash(password)
            conn = sqlite3.connect('plant_app.db', timeout=10.0)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, hashed_pw))
            conn.commit()
            conn.close()
            flash("Registration successful! Please log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
        except Exception as e:
            logger.error(f"Registration error: {e}")
            flash("Registration failed. Please try again.")
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Username and password are required.")
            return redirect(request.url)

        try:
            conn = sqlite3.connect('plant_app.db', timeout=10.0)
            c = conn.cursor()
            c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            conn.close()

            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['username'] = username
                flash("Login successful!")
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid credentials.")
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash("Login failed. Please try again.")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Check model status
        if MODEL_LOADING:
            flash("Model is loading. Please wait a moment and try again.")
            return redirect(request.url)
        
        if not MODEL_LOADED:
            flash("Model is not available. Please try again later.")
            return redirect(request.url)
        
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                predicted_class, confidence = predict_disease_safe(filepath)
                treatment_info = get_treatment_info(predicted_class, confidence)

                # Save to database with error handling
                try:
                    conn = sqlite3.connect('plant_app.db', timeout=10.0)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                        VALUES (?, ?, ?, ?)''',
                        (session['user_id'], filename, predicted_class, confidence))
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    logger.error(f"Database save error: {db_error}")

                return render_template('result.html', 
                                       filename=filename, 
                                       predicted_class=predicted_class, 
                                       confidence=round(confidence * 100, 2),
                                       treatment_info=treatment_info)
                                       
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash(f"Error processing image: {str(e)}")
        else:
            flash("Invalid file format or no file selected.")
    
    return render_template('predict.html', 
                         model_loaded=MODEL_LOADED,
                         model_loading=MODEL_LOADING)

@app.route('/model-status')
@login_required
def model_status():
    """Check model loading status"""
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "tensorflow_available": TENSORFLOW_AVAILABLE
    })

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = sqlite3.connect('plant_app.db', timeout=10.0)
        c = conn.cursor()
        
        c.execute('''
            SELECT filename, predicted_class, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        ''', (session['user_id'],))
        predictions = c.fetchall()
        
        c.execute('''
            SELECT 
                COUNT(*) as total_scans,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE user_id = ?
        ''', (session['user_id'],))
        stats = c.fetchone()
        conn.close()
        
        return render_template('dashboard.html', 
                            predictions=predictions,
                            stats=stats or (0, 0),
                            model_loaded=MODEL_LOADED,
                            model_loading=MODEL_LOADING)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', 
                            predictions=[],
                            stats=(0, 0),
                            model_loaded=MODEL_LOADED,
                            model_loading=MODEL_LOADING)

@app.route('/about')
@login_required  
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    if request.method == 'POST':
        flash("Thanks for reaching out! We'll get back to you.")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Graceful shutdown handling
def signal_handler(sig, frame):
    logger.info('Gracefully shutting down...')
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    logger.info("🚀 Starting Plant Disease Detection App...")
    logger.info("App ready immediately - Model loads in background")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)