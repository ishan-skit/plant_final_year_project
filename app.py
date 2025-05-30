# Flask Plant Disease Detection App - Render Optimized
import os
import sys

# CRITICAL: Set these environment variables FIRST
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import json
import logging
import sqlite3
import gc
import base64
from io import BytesIO
from datetime import datetime
from datetime import timedelta
from functools import wraps
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_caching import Cache
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv
from flask import send_from_directory
import threading
import queue
import time

# Load labels first
try:
    with open('model/labels_reverse.json', 'r') as f:
        LABELS_REVERSE = json.load(f)
except:
    LABELS_REVERSE = {}

# CRITICAL: Explicit layer registration to fix InputLayer deserialization
import tensorflow.keras.layers
tf.keras.layers.InputLayer = tf.keras.layers.InputLayer

# Configure logging for Render deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure TensorFlow for maximum stability and speed
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)  # Increased for better performance
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    # Enable mixed precision for faster inference
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
    logger.info("TensorFlow configured for optimized Render deployment")
except Exception as e:
    logger.warning(f"TensorFlow configuration warning: {e}")

# Configure Gemini API
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini AI configured successfully")

# Constants matching train_model.py EXACTLY
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'
IMG_SIZE = (128, 128)  # EXACT match with train_model.py
INPUT_SHAPE = (128, 128, 3)  # EXACT match with train_model.py

# Global prediction queue for async processing
prediction_queue = queue.Queue()
prediction_results = {}

# Initialize global model and label_dict
model = None
label_dict = {}

# Flask App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'render-plant-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB for faster processing

# Database configuration for Render
database_url = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Configure uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize cache with longer timeouts
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 3600  # 1 hour cache
})

# Google OAuth Configuration
oauth = OAuth(app)
google = None
if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
    google = oauth.register(
        name='google',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

# Configure session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Load treatments data
treatments_df = pd.DataFrame()
if os.path.exists(TREATMENTS_PATH):
    treatments_df = pd.read_csv(TREATMENTS_PATH)

def load_model_with_timeout():
    """Load model with timeout protection"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model loading timed out")
    
    # Set timeout for model loading (20 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    
    try:
        return load_model_safely()
    finally:
        signal.alarm(0)  # Cancel the alarm

def load_model_safely():
    """Enhanced model loading with timeout protection"""
    global model, label_dict
    
    max_retries = 2  # Reduced retries for faster startup
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[LOAD] Model loading attempt {attempt + 1}/{max_retries}...")
            
            if attempt == 0:
                # Method 1: Fast loading with compile=False
                logger.info("[METHOD 1] Fast loading with compile=False...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
                # Quick recompile
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
            else:
                # Method 2: Load with custom objects
                logger.info("[METHOD 2] Loading with custom objects...")
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                }
                
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']
                )
            
            # Quick test with smaller input
            logger.info("[TEST] Quick model test...")
            dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
            test_prediction = model.predict(dummy_input, verbose=0)
            
            logger.info(f"[SUCCESS] Model loaded! Output shape: {test_prediction.shape}")
            break
            
        except Exception as e:
            logger.warning(f"[RETRY] Method {attempt + 1} failed: {str(e)[:100]}...")
            
            if attempt == max_retries - 1:
                logger.error("[ERROR] Model loading failed!")
                return False
            
            model = None
            tf.keras.backend.clear_session()
            gc.collect()
    
    # Load labels
    try:
        with open(LABELS_PATH, 'r') as f:
            label_map = json.load(f)
        label_dict = {v: k for k, v in label_map.items()}
        logger.info(f"[SUCCESS] Loaded {len(label_dict)} class labels")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load labels: {e}")
        return False

# Database initialization
def init_db():
    """Initialize SQLite database for Render deployment"""
    try:
        with sqlite3.connect('plant_app.db', timeout=10) as conn:
            c = conn.cursor()
            
            # Users table
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
            c.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in c.fetchall()]
            if 'google_id' not in columns:
                c.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
            
            # Predictions table
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
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

init_db()

# Async prediction worker
def prediction_worker():
    """Background worker for processing predictions"""
    while True:
        try:
            task = prediction_queue.get(timeout=1)
            if task is None:
                break
                
            task_id, image_path = task
            
            # Process prediction
            start_time = time.time()
            predicted_class, confidence = predict_disease_fast(image_path)
            processing_time = time.time() - start_time
            
            # Store result
            prediction_results[task_id] = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'processing_time': processing_time,
                'status': 'completed'
            }
            
            prediction_queue.task_done()
            logger.info(f"Async prediction completed in {processing_time:.2f}s")
            
        except queue.Empty:
            continue
        except Exception as e:
            if 'task_id' in locals():
                prediction_results[task_id] = {
                    'error': str(e),
                    'status': 'failed'
                }
            logger.error(f"Prediction worker error: {e}")

# Start background worker
worker_thread = threading.Thread(target=prediction_worker, daemon=True)
worker_thread.start()

def getImmediateAction(disease_name):
    """Get immediate action recommendation based on disease name"""
    if not disease_name:
        return "Monitor plant condition"
    
    disease_lower = disease_name.lower()
    
    if 'healthy' in disease_lower:
        return "Continue current care routine"
    elif any(word in disease_lower for word in ['blight', 'rot', 'wilt']):
        return "Remove affected leaves immediately and isolate plant"
    elif any(word in disease_lower for word in ['rust', 'fungal', 'mold']):
        return "Apply fungicide and improve air circulation"
    elif any(word in disease_lower for word in ['spot', 'leaf spot']):
        return "Reduce watering frequency and remove affected areas"
    elif any(word in disease_lower for word in ['virus', 'mosaic']):
        return "Isolate plant immediately - virus may be contagious"
    elif any(word in disease_lower for word in ['pest', 'insect', 'aphid']):
        return "Apply insecticidal soap or neem oil treatment"
    else:
        return "Isolate plant and consult treatment guide"

app.jinja_env.globals.update(getImmediateAction=getImmediateAction)

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

@cache.memoize(timeout=3600)
def preprocess_image_cached(image_data):
    """Cached image preprocessing"""
    try:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_disease_fast(image_path):
    """Optimized prediction function"""
    try:
        # Read image data
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Use cached preprocessing
        img_array = preprocess_image_cached(image_data)
        if img_array is None:
            raise ValueError("Failed to preprocess image")

        # Fast prediction with reduced batch size
        predictions = model.predict(img_array, batch_size=1, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = LABELS_REVERSE.get(str(predicted_class_index), "Unknown")

        # Clean up memory
        del img_array, predictions
        gc.collect()

        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Fast prediction error: {e}")
        raise

def predict_disease(image_path):
    """Main prediction function with fallback"""
    try:
        return predict_disease_fast(image_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Unknown Disease", 0.0

@cache.memoize(timeout=1800)  # 30 minute cache
def get_ai_treatment_cached(disease_name):
    """Cached AI treatment to reduce API calls"""
    return get_ai_treatment(disease_name, 0.0)

def get_ai_treatment(disease_name, confidence_score=0.0):
    """Get ultra-light AI treatment response"""
    try:
        if not os.getenv("GEMINI_API_KEY"):
            return get_fallback_treatment(disease_name)

        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Brief treatment for plant disease: {disease_name}"

        response = model_ai.generate_content(prompt)

        if response and response.text:
            return {
                'disease': disease_name,
                'treatment': response.text.strip()[:500],  # Limit response size
                'prevention': 'See treatment',
                'organic_treatment': 'See treatment',
                'chemical_treatment': 'See treatment',
                'severity': 'Not specified',
                'source': 'Gemini AI'
            }
        else:
            return get_fallback_treatment(disease_name)

    except Exception as e:
        logger.error(f"AI treatment error: {e}")
        return get_fallback_treatment(disease_name)

def get_fallback_treatment(disease_name):
    """Fallback treatment when AI is unavailable"""
    return {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Apply appropriate treatment based on disease type. Consult agricultural extension office for specific recommendations.',
        'prevention': 'Maintain proper plant spacing, avoid overhead watering, remove infected parts.',
        'organic_treatment': 'Use neem oil, copper-based fungicides, or baking soda solution.',
        'chemical_treatment': 'Consult agricultural specialist for chemical treatments.',
        'severity': 'Requires assessment',
        'source': 'Fallback'
    }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment info with optimization"""
    CONFIDENCE_THRESHOLD = 0.7
    
    # Check CSV first for high confidence
    if not treatments_df.empty and confidence_score >= CONFIDENCE_THRESHOLD:
        treatment_row = treatments_df[treatments_df['disease'] == disease_name]
        if treatment_row.empty:
            treatment_row = treatments_df[treatments_df['disease'].str.contains(disease_name, case=False, na=False)]
        
        if not treatment_row.empty:
            row = treatment_row.iloc[0]
            return {
                'disease': row['disease'],
                'treatment': row['treatment'],
                'prevention': row['prevention'],
                'organic_treatment': row['organic_treatment'],
                'chemical_treatment': row['chemical_treatment'],
                'severity': row['severity'],
                'source': 'Database'
            }
    
    # Use cached AI treatment
    return get_ai_treatment_cached(disease_name)

# Health check for Render
@app.route('/health')
def health_check():
    """Health check endpoint for Render deployment"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": len(label_dict) > 0,
        "tf_version": tf.__version__,
        "timestamp": datetime.now().isoformat()
    }), 200

# Authentication routes (same as before but with timeout protection)
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
            with sqlite3.connect('plant_app.db', timeout=5) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                          (username, email, hashed_pw))
                conn.commit()
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
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Username and password are required.")
            return render_template('login.html')

        try:
            with sqlite3.connect('plant_app.db', timeout=5) as conn:
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
            logger.error(f"Login error: {e}")
            flash("Login failed. Please try again.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

# Main application routes
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        session_predictions = session.get('predictions', [])
        if not isinstance(session_predictions, list):
            session_predictions = []

        with sqlite3.connect('plant_app.db', timeout=5) as conn:
            c = conn.cursor()
            
            c.execute('''
                SELECT filename, predicted_class, confidence, strftime('%Y-%m-%d %H:%M:%S', created_at) as formatted_date
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            ''', (session['user_id'],))
            db_predictions = c.fetchall()
            
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

        all_predictions = session_predictions + [list(p) for p in db_predictions]
        
        seen_files = set()
        unique_predictions = []
        for pred in all_predictions:
            filename = pred[0] if len(pred) > 0 else ''
            if filename and filename not in seen_files:
                seen_files.add(filename)
                unique_predictions.append(pred)
                if len(unique_predictions) >= 10:
                    break

        avg_confidence_percent = round((stats[3] or 0) * 100, 1)
        
        return render_template('dashboard.html', 
                            predictions=unique_predictions,
                            total_scans=stats[0],
                            healthy_count=stats[1],
                            disease_count=stats[2],
                            avg_confidence=avg_confidence_percent)
                            
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return render_template('dashboard.html', 
                            predictions=[], 
                            total_scans=0,
                            healthy_count=0,
                            disease_count=0,
                            avg_confidence=0)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        file = request.files.get('image')

        if not file or file.filename == '' or not allowed_file(file.filename):
            flash("Please upload a valid image file (.jpg, .jpeg, .png, .gif).")
            return render_template('predict.html')

        try:
            # Save file with timeout protection
            filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Fast prediction with timeout
            start_time = time.time()
            predicted_class, confidence = predict_disease(filepath)
            processing_time = time.time() - start_time
            
            logger.info(f"Prediction completed in {processing_time:.2f}s")

            # Get treatment info
            treatment_info = get_treatment_info(predicted_class, confidence)

            # Save to database with timeout
            with sqlite3.connect('plant_app.db', timeout=5) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                    VALUES (?, ?, ?, ?)''',
                    (session['user_id'], filename, predicted_class, confidence))
                conn.commit()

            # Save to session
            if 'predictions' not in session:
                session['predictions'] = []

            prediction_entry = [
                filename,
                predicted_class,
                confidence,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]

            session['predictions'].insert(0, prediction_entry)
            session['predictions'] = session['predictions'][:5]  # Keep only 5 in session
            session.modified = True

            return render_template('result.html', 
                                   filename=filename, 
                                   predicted_class=predicted_class, 
                                   confidence=round(confidence * 100, 2),
                                   treatment_info=treatment_info,
                                   processing_time=round(processing_time, 2))

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash(f"Error processing image. Please try again.")
            return render_template('predict.html')

    return render_template('predict.html')

@app.route('/camera')
@login_required
def camera():
    """Real-time camera detection page"""
    return render_template('camera.html')

@app.route('/predict_camera', methods=['POST'])
@login_required  
def predict_camera():
    """Handle camera capture with timeout protection"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode and save image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"capture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Fast prediction
        predicted_class, confidence = predict_disease(filepath)
        treatment_info = get_treatment_info(predicted_class, confidence)
        
        # Save to database
        with sqlite3.connect('plant_app.db', timeout=5) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)''',
                (session['user_id'], filename, predicted_class, confidence))
            conn.commit()
        
        # Update session
        if 'predictions' not in session:
            session['predictions'] = []
        
        prediction_entry = [
            filename,
            predicted_class,
            confidence,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        session['predictions'].insert(0, prediction_entry)
        session['predictions'] = session['predictions'][:5]
        session.modified = True
        
        return jsonify({
            'success': True,
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'treatment_info': treatment_info
        })
        
    except Exception as e:
        logger.error(f"Camera prediction error: {e}")
        return jsonify({'error': 'Processing failed. Please try again.'}), 500

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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('errors/500.html'), 500

@app.errorhandler(408)
def timeout_error(error):
    logger.error(f"Request timeout: {error}")
    return jsonify({'error': 'Request timed out. Please try again.'}), 408

# Initialize model on startup with timeout protection
def initialize_app():
    """Initialize the application with timeout protection"""
    logger.info("Initializing optimized application...")
    init_db()
    
    try:
        if not load_model_with_timeout():
            logger.error("Failed to load model - using fallback mode")
        else:
            logger.info("Model loaded successfully!")
    except TimeoutError:
        logger.error("Model loading timed out - using fallback mode")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
    
    logger.info("Application initialization complete")

# Startup with timeout protection
if __name__ == '__main__':
    initialize_app()
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting optimized Flask app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Labels loaded: {len(label_dict)} classes")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
else:
    # For gunicorn deployment
    initialize_app()