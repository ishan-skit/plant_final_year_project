# Flask Plant Disease Detection App - Optimized for Render Deployment
import os
import json
import sys
import logging
import sqlite3
import gc
import base64
from io import BytesIO
from datetime import datetime
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

# Configure logging for Render deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# CRITICAL: Render CPU-only configuration matching train_model.py
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

# Configure TensorFlow for Render deployment
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logger.info("TensorFlow configured for Render CPU deployment")
except Exception as e:
    logger.warning(f"TensorFlow configuration warning: {e}")

# Configure Gemini API
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini AI configured successfully")

# Flask App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'render-plant-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for Render

# Database configuration for Render
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

# Constants matching train_model.py EXACTLY
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'
IMG_SIZE = (128, 128)  # EXACT match with train_model.py
INPUT_SHAPE = (128, 128, 3)  # EXACT match with train_model.py

# Global variables
model = None
label_dict = {}
treatments_df = pd.DataFrame()

# Render deployment model loading with retry mechanism
def load_model_safely():
    """Load model with error handling for Render deployment"""
    global model, label_dict
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading model from {MODEL_PATH} (attempt {attempt + 1})")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
                # Recompile with exact same settings as train_model.py
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    run_eagerly=False
                )
                
                logger.info("Model loaded and compiled successfully")
                break
            else:
                logger.error(f"Model file not found: {MODEL_PATH}")
                break
                
        except Exception as e:
            logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                tf.keras.backend.clear_session()
                gc.collect()
            else:
                model = None
    
    # Load labels
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                label_map = json.load(f)
            label_dict = {v: k for k, v in label_map.items()}
            logger.info(f"Labels loaded: {len(label_dict)} classes")
        else:
            logger.error(f"Labels file not found: {LABELS_PATH}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        label_dict = {}

# Load model and data on startup
load_model_safely()

# Load treatment data
try:
    if os.path.exists(TREATMENTS_PATH):
        treatments_df = pd.read_csv(TREATMENTS_PATH)
        logger.info(f"Treatment data loaded: {len(treatments_df)} treatments")
    else:
        logger.warning("Treatment CSV not found, using AI-only mode")
except Exception as e:
    logger.warning(f"Error loading treatment data: {e}")
    treatments_df = pd.DataFrame()

# Database initialization
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
    """Preprocess image EXACTLY as in train_model.py"""
    try:
        # Load and resize to exact training size
        img = Image.open(img_path).convert('RGB')
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
    """Predict disease with error handling for Render deployment"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        img_array = preprocess_image(img_path)
        if img_array is None:
            return "Image processing failed", 0.0
        
        # Predict with error handling
        preds = model.predict(img_array, verbose=0)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        class_name = label_dict.get(class_index, "Unknown")
        
        logger.info(f"Prediction: {class_name} (confidence: {confidence:.3f})")
        return class_name, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
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

# Main application routes
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            
            # Recent predictions
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
    
        return render_template('dashboard.html', 
                            predictions=predictions,
                            stats=stats)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash("Error loading dashboard.")
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

                # Save prediction
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
    """Real-time camera detection page"""
    return render_template('camera.html')

@app.route('/predict_camera', methods=['POST'])
@login_required  
def predict_camera():
    """Handle camera capture"""
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

# API endpoints
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
        "tf_version": tf.__version__,
        "input_shape": INPUT_SHAPE,
        "classes": list(label_dict.values())[:5] if label_dict else []
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('error.html', error="Internal server error"), 500

# Application startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Labels loaded: {len(label_dict)} classes")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)