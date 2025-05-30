# Flask Plant Disease Detection App - Optimized for Render Free Tier
import os
import sys

# CRITICAL: Set these environment variables FIRST - More aggressive for free tier
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'  # Critical for free tier
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'

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

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False
    print("TensorFlow not available")

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_caching import Cache
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging for Render deployment - More minimal
logging.basicConfig(
    level=logging.ERROR,  # Only errors to reduce memory usage
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure TensorFlow for Render free tier - AGGRESSIVE
if tf_available:
    try:
        # Force CPU and minimal memory usage
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        # Critical memory optimization for free tier
        tf.config.experimental.enable_op_determinism()
        
        logger.info("TensorFlow configured for Render free tier")
    except Exception as e:
        logger.warning(f"TensorFlow configuration warning: {e}")

# Configure Gemini API
if os.getenv("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Gemini AI configured successfully")
    except:
        logger.warning("Gemini AI configuration failed")

# Flask App Configuration - Optimized for free tier
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'render-plant-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB for free tier

# Database configuration for Render
database_url = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Configure uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize cache with memory optimization
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})

# Google OAuth Configuration
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
        logger.warning("Google OAuth configuration failed")

# Constants matching train_model.py EXACTLY
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'
IMG_SIZE = (128, 128)  # EXACT match with train_model.py
INPUT_SHAPE = (128, 128, 3)  # EXACT match with train_model.py

# Configure session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Global variables
model = None
label_dict = {}
treatments_df = pd.DataFrame()

def aggressive_memory_cleanup():
    """Aggressive memory cleanup for free tier"""
    try:
        gc.collect()
        if tf_available:
            tf.keras.backend.clear_session()
        return True
    except:
        return False

def load_model_with_timeout():
    """Load model with timeout and memory optimization"""
    global model, label_dict
    
    if not tf_available:
        logger.error("TensorFlow not available")
        return False
    
    try:
        logger.info("Loading model...")
        
        # Clear memory before loading
        aggressive_memory_cleanup()
        
        # Load model with minimal options for free tier
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False  # Don't compile to save memory
        )
        
        # Manually compile with minimal options
        model.compile(
            optimizer='adam',  # Use string instead of object to save memory
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False
        )
        
        # Test model quickly
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        test_prediction = model.predict(dummy_input, verbose=0)
        
        # Clean up test data immediately
        del dummy_input, test_prediction
        aggressive_memory_cleanup()
        
        logger.info("Model loaded successfully")
        
        # Load labels
        with open(LABELS_PATH, 'r') as f:
            label_map = json.load(f)
        label_dict = {v: k for k, v in label_map.items()}
        
        logger.info(f"Loaded {len(label_dict)} class labels")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model = None
        label_dict = {}
        aggressive_memory_cleanup()
        return False

# Database initialization - Simplified
def init_db():
    """Initialize SQLite database for Render deployment"""
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
            try:
                c.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
            except:
                pass  # Column might already exist
            
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
    """Get immediate action recommendation based on disease name"""
    if not disease_name:
        return "Monitor plant condition"
    
    disease_lower = disease_name.lower()
    
    # Define immediate actions for different diseases
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

# Make the function available in Jinja templates
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

def preprocess_image(img_path):
    """Preprocess image EXACTLY as in train_model.py - OPTIMIZED"""
    try:
        # Load and resize to exact training size
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to array and normalize EXACTLY as training
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # EXACT normalization as train_model.py
            
            return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_disease(img_path):
    """Predict disease with AGGRESSIVE optimization for free tier"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Clear memory before prediction
        aggressive_memory_cleanup()
        
        # Preprocess image
        img_array = preprocess_image(img_path)
        if img_array is None:
            return "Image processing failed", 0.0
        
        # Predict with minimal verbosity
        preds = model.predict(img_array, verbose=0, batch_size=1)
        
        # Get results quickly
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        class_name = label_dict.get(class_index, "Unknown")
        
        # Clean up immediately
        del img_array, preds
        aggressive_memory_cleanup()
        
        logger.info(f"Prediction: {class_name} (confidence: {confidence:.3f})")
        return class_name, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        aggressive_memory_cleanup()
        return f"Prediction failed: {str(e)}", 0.0

def get_ai_treatment(disease_name, confidence_score=0.0):
    """Get AI treatment with fallback for Render deployment"""
    try:
        if not os.getenv("GEMINI_API_KEY"):
            return get_fallback_treatment(disease_name)
        
        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        As a plant pathology expert, provide treatment for: "{disease_name}"
        
        Include:
        1. Disease overview
        2. Treatment options (chemical & organic)
        3. Prevention measures
        4. Severity assessment
        5. Monitoring advice
        
        Keep response practical and concise for farmers.
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
    """Fallback treatment when AI is unavailable"""
    return {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Apply general fungicide if fungal disease, or contact local agricultural extension office for specific treatment recommendations.',
        'prevention': 'Maintain proper plant spacing, avoid overhead watering, remove infected plant parts, and ensure good air circulation.',
        'organic_treatment': 'Use neem oil, copper-based fungicides, or baking soda solution as organic alternatives.',
        'chemical_treatment': 'Consult with agricultural specialist for appropriate chemical treatments based on local regulations.',
        'severity': 'Requires assessment',
        'source': 'Fallback'
    }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment info with CSV fallback and AI enhancement"""
    CONFIDENCE_THRESHOLD = 0.7
    
    # Check CSV first if available and confidence is high
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
    
    # Use AI for unknown diseases or low confidence
    return get_ai_treatment(disease_name, confidence_score)

# Health check for Render
@app.route('/health')
def health_check():
    """Health check endpoint for Render deployment"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": len(label_dict) > 0,
        "tf_version": tf.__version__ if tf_available else "N/A"
    }), 200

# Authentication routes (keeping existing code but simplified error handling)
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

# Main application routes
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get session predictions for immediate updates
        session_predictions = session.get('predictions', [])
        
        # Ensure session predictions is always a list
        if not isinstance(session_predictions, list):
            session_predictions = []
        
        # Get database predictions for historical data
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            
            # Recent predictions from database
            c.execute('''
                SELECT filename, predicted_class, confidence, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            ''', (session['user_id'],))
            db_predictions = c.fetchall()
            
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
        
        # Combine session and database predictions (session first for recent activity)
        all_predictions = session_predictions + [list(p) for p in db_predictions]
        
        # Remove duplicates based on filename and limit to 10
        seen_files = set()
        unique_predictions = []
        for pred in all_predictions:
            filename = pred[0] if len(pred) > 0 else ''
            if filename not in seen_files and len(unique_predictions) < 10:
                seen_files.add(filename)
                unique_predictions.append(pred)
        
        return render_template('dashboard.html', 
                            predictions=unique_predictions,
                            stats=stats)
                            
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        # Return dashboard with empty data if there's an error
        return render_template('dashboard.html', 
                            predictions=[], 
                            stats=(0, 0, 0, 0))

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

                # Clear memory before prediction
                aggressive_memory_cleanup()

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

                # ALSO save to session for immediate dashboard updates
                if 'predictions' not in session:
                    session['predictions'] = []
                
                # Create prediction entry with timestamp
                prediction_entry = [
                    filename,  # image filename
                    predicted_class,  # disease name
                    confidence,  # confidence score (0-1)
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # timestamp
                ]
                
                # Add to beginning of list (most recent first)
                session['predictions'].insert(0, prediction_entry)
                
                # Keep only last 5 predictions in session to save memory
                session['predictions'] = session['predictions'][:5]
                
                # Make sure session is saved
                session.modified = True
                
                # Clean up after prediction
                aggressive_memory_cleanup()

                return render_template('result.html', 
                                    filename=filename, 
                                    predicted_class=predicted_class, 
                                    confidence=round(confidence * 100, 2),
                                    treatment_info=treatment_info)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                aggressive_memory_cleanup()
                flash("Error processing image. Please try again.")
        else:
            flash("Invalid file format or no file selected.")
            
    return render_template('predict.html')
                     
from flask import send_from_directory  

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files dynamically"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
@login_required
def camera():
    """Real-time camera detection page"""
    return render_template('camera.html')

@app.route('/predict_camera', methods=['POST'])
@login_required  
def predict_camera():
    """Handle camera capture - OPTIMIZED"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Clear memory before processing
        aggressive_memory_cleanup()
        
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
        
        # ALSO save to session for immediate dashboard updates
        if 'predictions' not in session:
            session['predictions'] = []
        
        prediction_entry = [
            filename,
            predicted_class,
            confidence,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        session['predictions'].insert(0, prediction_entry)
        session['predictions'] = session['predictions'][:5]  # Reduced for memory
        session.modified = True
        
        # Clean up
        aggressive_memory_cleanup()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'treatment_info': treatment_info
        })
        
    except Exception as e:
        logger.error(f"Camera prediction error: {e}")
        aggressive_memory_cleanup()
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

# API endpoints - Simplified
@app.route('/api/ai_treatment', methods=['POST'])
@login_required
def api_ai_treatment():
    try:
        data = request.get_json()
        disease = data.get('disease')
        confidence = data.get('confidence', 0.0)
        
        if not disease:
            return jsonify({"error": "Disease name is required"}), 400
        
        treatment_info = get_ai_treatment(disease, confidence)
        return jsonify(treatment_info)
    except Exception as e:
        logger.error(f"AI treatment API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test_model')
@login_required
def test_model():
    """Test endpoint for model functionality"""
    return jsonify({
        "model_loaded": model is not None,
        "labels_count": len(label_dict),
        "tf_version": tf.__version__ if tf_available else "N/A",
        "input_shape": INPUT_SHAPE,
        "classes": list(label_dict.values())[:5] if label_dict else []
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    aggressive_memory_cleanup()
    return render_template('errors/500.html'), 500

@app.errorhandler(413)
def too_large(error):
    flash("File too large. Please upload a smaller image.")
    return redirect(url_for('predict'))

# Initialize model on startup with timeout
def initialize_app():
    """Initialize the application with timeout"""
    logger.info("Initializing application...")
    init_db()
    
    # Try to load model with multiple attempts
    for attempt in range(3):
        try:
            if load_model_with_timeout():
                logger.info("Application initialization complete")
                return
            else:
                logger.warning(f"Model loading attempt {attempt + 1} failed")
                aggressive_memory_cleanup()
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            aggressive_memory_cleanup()
    
    logger.error("Failed to initialize application after multiple attempts")

initialize_app()

# Application startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Labels loaded: {len(label_dict)} classes")

    app.run(host='0.0.0.0', port=port, debug=debug_mode)