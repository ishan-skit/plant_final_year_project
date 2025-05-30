# Flask Plant Disease Detection App - Ultra-Optimized for Render Free Tier
import os
import sys

# CRITICAL: Set these environment variables FIRST - Ultra-aggressive for free tier
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_ENABLE_RESOURCE_VARIABLES'] = 'false'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import json
import logging
import sqlite3
import gc
import base64
from io import BytesIO
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import pandas as pd

# Import TensorFlow with aggressive memory management
try:
    import tensorflow as tf
    # Force immediate memory optimization
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    tf_available = True
except ImportError:
    tf_available = False

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_caching import Cache
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv

# Ultra-minimal logging for free tier
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini API
if os.getenv("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    except:
        pass

# Flask App Configuration - Ultra-optimized
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'render-plant-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # Reduced to 4MB

# Database configuration
database_url = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

# Configure uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ultra-minimal cache
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 60})

# OAuth configuration
oauth = OAuth(app)
google = None
if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
    try:
        google = oauth.register(
            name='google',
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
    except:
        pass

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

# Session config
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=12)  # Reduced

# Global variables - Keep minimal
model = None
label_dict = {}
model_loaded = False
prediction_cache = {}  # Simple in-memory cache

def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    try:
        gc.collect()
        if tf_available and 'tf' in globals():
            tf.keras.backend.clear_session()
        # Clear prediction cache if it gets too large
        global prediction_cache
        if len(prediction_cache) > 10:
            prediction_cache.clear()
        return True
    except:
        return False

def load_model_lazy():
    """Lazy load model only when needed"""
    global model, label_dict, model_loaded
    
    if model_loaded:
        return True
        
    if not tf_available:
        return False
    
    try:
        ultra_memory_cleanup()
        
        # Load model with minimal memory footprint
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Compile with minimal options
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            run_eagerly=False
        )
        
        # Load labels
        with open(LABELS_PATH, 'r') as f:
            label_map = json.load(f)
        label_dict = {v: k for k, v in label_map.items()}
        
        model_loaded = True
        logger.info(f"Model loaded with {len(label_dict)} classes")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model = None
        label_dict = {}
        model_loaded = False
        ultra_memory_cleanup()
        return False

# Database initialization
def init_db():
    try:
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT,
                    google_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add google_id column if missing
            try:
                c.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
            except:
                pass
            
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
            
    except Exception as e:
        logger.error(f"Database init error: {e}")

init_db()

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image_optimized(img_path):
    """Ultra-optimized image preprocessing"""
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_disease_optimized(img_path):
    """Ultra-optimized prediction with caching"""
    global prediction_cache
    
    # Check cache first
    cache_key = f"{img_path}_{os.path.getmtime(img_path)}"
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]
    
    # Lazy load model
    if not load_model_lazy():
        return "Model not available", 0.0
    
    try:
        ultra_memory_cleanup()
        
        img_array = preprocess_image_optimized(img_path)
        if img_array is None:
            return "Image processing failed", 0.0
        
        # Predict with batch_size=1 and minimal options
        with tf.device('/CPU:0'):  # Force CPU
            preds = model.predict(img_array, verbose=0, batch_size=1, use_multiprocessing=False)
        
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        class_name = label_dict.get(class_index, "Unknown")
        
        # Cache result
        result = (class_name, confidence)
        prediction_cache[cache_key] = result
        
        # Cleanup
        del img_array, preds
        ultra_memory_cleanup()
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ultra_memory_cleanup()
        return f"Prediction failed", 0.0

def get_fallback_treatment(disease_name):
    """Lightweight fallback treatment"""
    return {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Apply appropriate treatment based on disease type. Consult agricultural extension office for specific recommendations.',
        'prevention': 'Maintain proper plant spacing, avoid overhead watering, remove infected parts, ensure good air circulation.',
        'organic_treatment': 'Use neem oil, copper-based fungicides, or baking soda solution.',
        'chemical_treatment': 'Consult agricultural specialist for chemical treatments.',
        'severity': 'Assessment needed',
        'source': 'Fallback'
    }

def get_ai_treatment_optimized(disease_name, confidence_score=0.0):
    """Optimized AI treatment with timeout"""
    try:
        if not os.getenv("GEMINI_API_KEY") or confidence_score < 0.6:
            return get_fallback_treatment(disease_name)
        
        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Provide concise treatment for plant disease: {disease_name}. Include treatment, prevention, and monitoring advice in under 200 words."
        
        response = model_ai.generate_content(prompt)
        
        if response and response.text:
            return {
                'disease': disease_name,
                'treatment': response.text.strip()[:500],  # Limit response size
                'prevention': 'Included above',
                'organic_treatment': 'Included above',
                'chemical_treatment': 'Included above',
                'severity': 'AI assessed',
                'source': 'Gemini AI'
            }
    except:
        pass
    
    return get_fallback_treatment(disease_name)

# Routes
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "memory_usage": "optimized"
    }), 200

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not all([username, email, password]):
            flash("All fields are required!")
            return render_template('register.html')

        try:
            hashed_pw = generate_password_hash(password)
            with sqlite3.connect('plant_app.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                          (username, email, hashed_pw))
                conn.commit()
            flash("Registration successful! Please log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
        except Exception as e:
            flash("Registration failed. Please try again.")
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Username and password are required.")
            return render_template('login.html')

        try:
            with sqlite3.connect('plant_app.db') as conn:
                c = conn.cursor()
                c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
                user = c.fetchone()

            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['username'] = username
                flash("Login successful!")
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid credentials.")
        except Exception as e:
            flash("Login failed. Please try again.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    ultra_memory_cleanup()  # Clean up on logout
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            
            c.execute('''
                SELECT filename, predicted_class, confidence, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            ''', (session['user_id'],))
            predictions = c.fetchall()
            
            c.execute('''
                SELECT 
                    COUNT(*) as total_scans,
                    SUM(CASE WHEN predicted_class LIKE '%healthy%' THEN 1 ELSE 0 END) as healthy_count,
                    SUM(CASE WHEN predicted_class NOT LIKE '%healthy%' THEN 1 ELSE 0 END) as disease_count,
                    AVG(confidence) as avg_confidence
                FROM predictions
                WHERE user_id = ?
            ''', (session['user_id'],))
            stats = c.fetchone() or (0, 0, 0, 0)
        
        return render_template('dashboard.html', predictions=predictions, stats=stats)
                            
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', predictions=[], stats=(0, 0, 0, 0))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                # Clean up before processing
                ultra_memory_cleanup()
                
                filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                predicted_class, confidence = predict_disease_optimized(filepath)
                treatment_info = get_ai_treatment_optimized(predicted_class, confidence)

                # Save to database
                with sqlite3.connect('plant_app.db') as conn:
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                        VALUES (?, ?, ?, ?)''',
                        (session['user_id'], filename, predicted_class, confidence))
                    conn.commit()

                # Clean up after processing
                ultra_memory_cleanup()

                return render_template('result.html', 
                                    filename=filename, 
                                    predicted_class=predicted_class, 
                                    confidence=round(confidence * 100, 2),
                                    treatment_info=treatment_info)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                ultra_memory_cleanup()
                flash("Error processing image. Please try again.")
        else:
            flash("Invalid file format or no file selected.")
            
    return render_template('predict.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')

@app.route('/predict_camera', methods=['POST'])
@login_required  
def predict_camera():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        ultra_memory_cleanup()
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"capture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        predicted_class, confidence = predict_disease_optimized(filepath)
        treatment_info = get_ai_treatment_optimized(predicted_class, confidence)
        
        # Save to database
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)''',
                (session['user_id'], filename, predicted_class, confidence))
            conn.commit()
        
        ultra_memory_cleanup()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'treatment_info': treatment_info
        })
        
    except Exception as e:
        logger.error(f"Camera prediction error: {e}")
        ultra_memory_cleanup()
        return jsonify({'error': str(e)}), 500

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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    ultra_memory_cleanup()
    return render_template('errors/500.html'), 500

@app.errorhandler(413)
def too_large(error):
    flash("File too large. Please upload a smaller image.")
    return redirect(url_for('predict'))

# Don't load model on startup - use lazy loading instead
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)