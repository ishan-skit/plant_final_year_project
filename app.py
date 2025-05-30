# Flask Plant Disease Detection App - Memory Optimized for Render
import os
import sys
import gc
import psutil
import threading
from datetime import datetime

# CRITICAL: Set environment variables FIRST before ANY TF imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_MEMORY_GROWTH'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit BLAS threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

import json
import logging
import sqlite3
import base64
from io import BytesIO
from datetime import timedelta
from functools import wraps
import numpy as np
import pandas as pd

# Import TensorFlow with memory optimization
import tensorflow as tf

# CRITICAL: Configure TensorFlow IMMEDIATELY after import
def configure_tensorflow():
    """Configure TensorFlow for minimal memory usage"""
    try:
        # Disable GPU completely
        tf.config.set_visible_devices([], 'GPU')
        
        # Limit CPU threads
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Disable experimental features
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        # Set memory growth for CPU (if supported)
        try:
            cpus = tf.config.list_physical_devices('CPU')
            if cpus:
                tf.config.experimental.set_memory_growth(cpus[0], True)
        except:
            pass
            
        return True
    except Exception as e:
        print(f"TensorFlow config warning: {e}")
        return False

configure_tensorflow()

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_caching import Cache
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini AI configured successfully")

# Flask App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'render-plant-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB

# Database configuration
database_url = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Configure uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Removed gif to save memory
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

# Configure session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Global variables for lazy loading
model = None
label_dict = {}
model_lock = threading.Lock()
last_prediction_time = None

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_memory():
    """Aggressive memory cleanup"""
    try:
        gc.collect()
        tf.keras.backend.clear_session()
    except:
        pass

def load_model_lazy():
    """
    Enhanced lazy load model with multiple fallback strategies for version compatibility
    """
    global model, label_dict, last_prediction_time
    
    with model_lock:
        if model is not None:
            last_prediction_time = datetime.now()
            return True
            
        try:
            logger.info(f"Loading model... Current memory: {get_memory_usage():.1f}MB")
            
            # Clear any existing TF session
            tf.keras.backend.clear_session()
            
            # Strategy 1: Try normal loading first
            try:
                logger.info("Attempting Strategy 1: Normal model loading")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
                # Recompile with compatible optimizer
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    run_eagerly=False
                )
                logger.info("✅ Strategy 1 successful")
                
            except Exception as e1:
                logger.warning(f"Strategy 1 failed: {e1}")
                
                # Strategy 2: Load with custom objects and different compile options
                try:
                    logger.info("Attempting Strategy 2: Custom objects loading")
                    model = tf.keras.models.load_model(
                        MODEL_PATH, 
                        compile=False,
                        custom_objects={}
                    )
                    
                    # Simple recompilation
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    logger.info("✅ Strategy 2 successful")
                    
                except Exception as e2:
                    logger.warning(f"Strategy 2 failed: {e2}")
                    
                    # Strategy 3: Try loading SavedModel format if available
                    try:
                        logger.info("Attempting Strategy 3: SavedModel format")
                        saved_model_path = MODEL_PATH.replace('.h5', '_savedmodel')
                        if os.path.exists(saved_model_path):
                            model = tf.keras.models.load_model(saved_model_path, compile=False)
                            model.compile(
                                optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy']
                            )
                            logger.info("✅ Strategy 3 successful")
                        else:
                            raise Exception("SavedModel format not available")
                            
                    except Exception as e3:
                        logger.error(f"All loading strategies failed: {e1}, {e2}, {e3}")
                        model = None
                        return False
            
            # Load labels
            with open(LABELS_PATH, 'r') as f:
                label_map = json.load(f)
            label_dict = {v: k for k, v in label_map.items()}
            
            # Test the model with a dummy prediction
            try:
                dummy_input = np.random.random((1, 128, 128, 3))
                test_pred = model.predict(dummy_input, batch_size=1, verbose=0)
                logger.info(f"Model test successful: output shape {test_pred.shape}")
                del dummy_input, test_pred  # Clean up test variables
            except Exception as test_error:
                logger.error(f"Model test failed: {test_error}")
                model = None
                return False
            
            last_prediction_time = datetime.now()
            logger.info(f"Model loaded successfully! Memory: {get_memory_usage():.1f}MB")
            logger.info(f"Model has {len(label_dict)} classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
            label_dict = {}
            cleanup_memory()
            return False
 

def unload_model():
    """Unload model to free memory"""
    global model, label_dict
    
    with model_lock:
        if model is not None:
            logger.info(f"Unloading model... Current memory: {get_memory_usage():.1f}MB")
            del model
            model = None
            label_dict = {}
            cleanup_memory()
            logger.info(f"Model unloaded. Memory: {get_memory_usage():.1f}MB")

def should_unload_model():
    """Check if model should be unloaded to save memory"""
    if model is None or last_prediction_time is None:
        return False
    
    # Unload if not used for 10 minutes
    time_since_last_use = datetime.now() - last_prediction_time
    return time_since_last_use.total_seconds() > 600

# Background task to manage memory
def memory_manager():
    """Background thread to manage memory usage"""
    import time
    while True:
        try:
            time.sleep(60)  # Check every minute
            
            current_memory = get_memory_usage()
            
            # If memory usage is high or model hasn't been used recently
            if current_memory > 400 or should_unload_model():  # 400MB threshold
                if model is not None:
                    logger.info(f"High memory usage ({current_memory:.1f}MB), unloading model")
                    unload_model()
                    
        except Exception as e:
            logger.error(f"Memory manager error: {e}")

# Start memory manager thread
memory_thread = threading.Thread(target=memory_manager, daemon=True)
memory_thread.start()

def init_db():
    """Initialize database"""
    try:
        with sqlite3.connect('plant_app.db') as conn:
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

def getImmediateAction(disease_name):
    """Get immediate action recommendation"""
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

def preprocess_image(img_path):
    """Memory-efficient image preprocessing"""
    try:
        # Load and resize
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to array efficiently
            img_array = np.array(img, dtype=np.float32)
            
        # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None


def predict_disease(img_path):
    """
    Enhanced disease prediction with better error handling
    """
    try:
        # Load model if needed
        if not load_model_lazy():
            return "Model loading failed - version compatibility issue", 0.0
        
        # Preprocess image
        img_array = preprocess_image(img_path)
        if img_array is None:
            return "Image processing failed", 0.0
        
        # Predict with enhanced error handling
        with model_lock:
            try:
                # Try batch prediction first
                preds = model.predict(img_array, batch_size=1, verbose=0)
            except Exception as pred_error:
                logger.warning(f"Batch prediction failed: {pred_error}")
                try:
                    # Fallback to single prediction
                    preds = model(img_array, training=False)
                    if hasattr(preds, 'numpy'):
                        preds = preds.numpy()
                except Exception as single_pred_error:
                    logger.error(f"Single prediction also failed: {single_pred_error}")
                    return "Prediction failed - model compatibility issue", 0.0
            
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        class_name = label_dict.get(class_index, "Unknown")
        
        # Cleanup
        del img_array, preds
        cleanup_memory()
        
        logger.info(f"Prediction: {class_name} (confidence: {confidence:.3f})")
        logger.info(f"Memory after prediction: {get_memory_usage():.1f}MB")
        
        return class_name, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        cleanup_memory()
        return f"Prediction failed: {str(e)}", 0.0

def get_ai_treatment(disease_name, confidence_score=0.0):
    """Get AI treatment with fallback"""
    try:
        if not os.getenv("GEMINI_API_KEY"):
            return get_fallback_treatment(disease_name)
        
        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        As a plant pathology expert, provide concise treatment for: "{disease_name}"
        
        Include:
        1. Brief disease overview
        2. Primary treatment (chemical & organic)
        3. Key prevention measures
        4. Severity level
        
        Keep response under 200 words for mobile users.
        """
        
        response = model_ai.generate_content(prompt)
        
        if response and response.text:
            return {
                'disease': disease_name,
                'treatment': response.text.strip(),
                'prevention': 'Included in AI response above',
                'organic_treatment': 'Included in AI response above',
                'chemical_treatment': 'Included in AI response above',
                'severity': 'AI assessed',
                'source': 'Gemini AI'
            }
        else:
            return get_fallback_treatment(disease_name)
            
    except Exception as e:
        logger.error(f"AI treatment error: {e}")
        return get_fallback_treatment(disease_name)

def get_fallback_treatment(disease_name):
    """Fallback treatment when AI unavailable"""
    return {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Apply appropriate fungicide if fungal, or contact agricultural extension office.',
        'prevention': 'Maintain proper spacing, avoid overhead watering, remove infected parts.',
        'organic_treatment': 'Use neem oil, copper fungicides, or baking soda solution.',
        'chemical_treatment': 'Consult agricultural specialist for appropriate treatments.',
        'severity': 'Requires assessment',
        'source': 'Fallback'
    }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment information"""
    return get_ai_treatment(disease_name, confidence_score)

# Health check
@app.route('/health')
def health_check():
    """Health check for Render"""
    memory_usage = get_memory_usage()
    return jsonify({
        "status": "healthy",
        "memory_mb": round(memory_usage, 1),
        "model_loaded": model is not None,
        "labels_loaded": len(label_dict) > 0,
        "tf_version": tf.__version__
    }), 200

# Authentication routes
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
            logger.error(f"Login error: {e}")
            flash("Login failed. Please try again.")
            
    return render_template('login.html')

@app.route('/auth/google')
def google_login():
    if not google:
        flash("Google login not configured.")
        return redirect(url_for('login'))
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def google_callback():
    if not google:
        flash("Google login not configured.")
        return redirect(url_for('login'))
        
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if user_info:
            google_id = user_info['sub']
            email = user_info['email']
            name = user_info['name']
            
            with sqlite3.connect('plant_app.db') as conn:
                c = conn.cursor()
                c.execute("SELECT id, username FROM users WHERE google_id = ? OR email = ?", (google_id, email))
                user = c.fetchone()
                
                if user:
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                else:
                    username = name.replace(' ', '_').lower()[:50]
                    try:
                        c.execute("INSERT INTO users (username, email, google_id) VALUES (?, ?, ?)",
                                  (username, email, google_id))
                        conn.commit()
                        session['user_id'] = c.lastrowid
                        session['username'] = username
                    except sqlite3.IntegrityError:
                        username = f"{username}_{datetime.now().strftime('%Y%m%d')}"
                        c.execute("INSERT INTO users (username, email, google_id) VALUES (?, ?, ?)",
                                  (username, email, google_id))
                        conn.commit()
                        session['user_id'] = c.lastrowid
                        session['username'] = username
            
            flash("Login successful!")
            return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"OAuth error: {e}")
        flash("Google login failed. Please try again.")
    
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get recent predictions
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            
            c.execute('''
                SELECT filename, predicted_class, confidence, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            ''', (session['user_id'],))
            predictions = c.fetchall()
            
            # Statistics
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
                filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                predicted_class, confidence = predict_disease(filepath)
                treatment_info = get_treatment_info(predicted_class, confidence)

                # Save to database
                with sqlite3.connect('plant_app.db') as conn:
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                        VALUES (?, ?, ?, ?)''',
                        (session['user_id'], filename, predicted_class, confidence))
                    conn.commit()

                return render_template('result.html', 
                                    filename=filename, 
                                    predicted_class=predicted_class, 
                                    confidence=round(confidence * 100, 2),
                                    treatment_info=treatment_info)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash("Error processing image. Please try again.")
        else:
            flash("Invalid file format or no file selected.")
            
    return render_template('predict.html')

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
        
        # Decode image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"capture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Predict
        predicted_class, confidence = predict_disease(filepath)
        treatment_info = get_treatment_info(predicted_class, confidence)
        
        # Save to database
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)''',
                (session['user_id'], filename, predicted_class, confidence))
            conn.commit()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'treatment_info': treatment_info
        })
        
    except Exception as e:
        logger.error(f"Camera prediction error: {e}")
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
    logger.error(f"Internal error: {error}")
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Initial memory usage: {get_memory_usage():.1f}MB")

    app.run(host='0.0.0.0', port=port, debug=debug_mode)