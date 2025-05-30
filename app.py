# Flask Plant Disease Detection App - Optimized for Render Deployment
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

# Configure TensorFlow for stability
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
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
import os
print("[DEBUG] Model exists:", os.path.exists(MODEL_PATH))

LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'
IMG_SIZE = (128, 128)  # EXACT match with train_model.py
INPUT_SHAPE = (128, 128, 3)  # EXACT match with train_model.py

# Configure session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = True # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Global variables
model = None
label_dict = {}
treatments_df = pd.DataFrame()

def load_model_safely():
    """Enhanced model loading with multiple fallback strategies for InputLayer issues"""
    global model, label_dict
    
    max_retries = 4
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[LOAD] Model loading attempt {attempt + 1}/{max_retries}...")
            
            if attempt == 0:
                # Method 1: Load with custom objects and compile=False
                logger.info("[METHOD 1] Loading with custom objects...")
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                }
                
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Recompile manually
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
            elif attempt == 1:
                # Method 2: Load with compile=False only
                logger.info("[METHOD 2] Loading with compile=False...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
            elif attempt == 2:
                # Method 3: Load and rebuild input layer
                logger.info("[METHOD 3] Loading and rebuilding input layer...")
                
                # Load without compiling
                temp_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                
                # Create new input layer
                new_input = tf.keras.layers.Input(shape=(128, 128, 3), name='input_layer')
                
                # Get all layers except input
                layers_to_use = []
                for i, layer in enumerate(temp_model.layers):
                    if not isinstance(layer, tf.keras.layers.InputLayer):
                        layers_to_use.append(layer)
                
                # Build new model
                x = new_input
                for layer in layers_to_use:
                    x = layer(x)
                
                model = tf.keras.Model(inputs=new_input, outputs=x)
                
                # Copy weights
                for old_layer, new_layer in zip(temp_model.layers, model.layers):
                    if old_layer.get_weights():
                        new_layer.set_weights(old_layer.get_weights())
                
                # Compile
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                del temp_model
                
            elif attempt == 3:
                # Method 4: Try loading architecture and weights separately (if they exist)
                logger.info("[METHOD 4] Trying architecture + weights fallback...")
                
                arch_path = 'model/model_architecture.json'
                weights_path = 'model/model_weights.h5'
                
                if os.path.exists(arch_path) and os.path.exists(weights_path):
                    with open(arch_path, 'r') as f:
                        model_json = f.read()
                    
                    model = tf.keras.models.model_from_json(model_json)
                    model.load_weights(weights_path)
                    
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    raise FileNotFoundError("Architecture/weights files not found")
            
            # Test the loaded model
            logger.info("[TEST] Testing loaded model...")
            dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
            test_prediction = model.predict(dummy_input, verbose=0)
            
            logger.info(f"[SUCCESS] Model loaded successfully! Output shape: {test_prediction.shape}")
            logger.info(f"[SUCCESS] Method {attempt + 1} worked!")
            break
            
        except Exception as e:
            logger.warning(f"[RETRY] Method {attempt + 1} failed: {str(e)[:100]}...")
            
            if attempt == max_retries - 1:
                logger.error("[ERROR] All model loading methods failed!")
                logger.error(f"Final error: {e}")
                return False
            
            # Clear any partially loaded model
            model = None
            tf.keras.backend.clear_session()
    
    # Load labels
    try:
        with open(LABELS_PATH, 'r') as f:
            label_map = json.load(f)
        label_dict = {v: k for k, v in label_map.items()}
        logger.info(f"[SUCCESS] Loaded {len(label_dict)} class labels")
        
        # Verify model output matches label count
        if test_prediction.shape[1] != len(label_dict):
            logger.warning(f"[WARNING] Model output ({test_prediction.shape[1]}) doesn't match label count ({len(label_dict)})")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load labels: {e}")
        return False

def diagnose_model_issue():
    """Diagnose model file to understand the issue"""
    try:
        logger.info("=== MODEL DIAGNOSIS ===")
        logger.info(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        logger.info(f"Labels file exists: {os.path.exists(LABELS_PATH)}")
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            logger.info(f"Model file size: {file_size:,} bytes")
            
            # Try to peek into the HDF5 structure
            try:
                import h5py
                with h5py.File(MODEL_PATH, 'r') as f:
                    logger.info("Model file structure (top level):")
                    for key in f.keys():
                        logger.info(f"  - {key}")
            except Exception as e:
                logger.info(f"Could not read HDF5 structure: {e}")
        
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"Python version: {sys.version}")
        logger.info("=== END DIAGNOSIS ===")
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")

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
                SELECT filename, predicted_class, confidence, strftime('%Y-%m-%d %H:%M:%S', created_at) as formatted_date
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 20
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
        
        # Remove duplicates based on filename and limit to 20
        seen_files = set()
        unique_predictions = []
        for pred in all_predictions:
            filename = pred[0] if len(pred) > 0 else ''
            if filename and filename not in seen_files:
                seen_files.add(filename)
                unique_predictions.append(pred)
                if len(unique_predictions) >= 20:
                    break
        
        # Sort by timestamp if available (most recent first)
        try:
            unique_predictions = sorted(unique_predictions, 
                                     key=lambda x: x[3] if len(x) > 3 else '', 
                                     reverse=True)
        except:
            pass  # If sorting fails, use original order
        
        app.logger.info(f"Dashboard loaded with {len(unique_predictions)} predictions")
        
        # Calculate confidence percentage for display
        avg_confidence_percent = round((stats[3] or 0) * 100, 1)
        
        return render_template('dashboard.html', 
                            predictions=unique_predictions,
                            total_scans=stats[0],
                            healthy_count=stats[1],
                            disease_count=stats[2],
                            avg_confidence=avg_confidence_percent)
                            
    except Exception as e:
        app.logger.error(f"Dashboard error: {str(e)}")
        # Return dashboard with empty data if there's an error
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
                
                # Keep only last 10 predictions in session to avoid bloat
                session['predictions'] = session['predictions'][:10]
                
                # Make sure session is saved
                session.modified = True
                
                app.logger.info(f"Saved prediction: {predicted_class} with {confidence:.3f} confidence")

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
        session['predictions'] = session['predictions'][:10]
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

@app.route('/save_detection', methods=['POST'])
@login_required
def save_detection():
    try:
        data = request.get_json()
        predicted_class = data.get('predicted_class')
        confidence = data.get('confidence')
        image_data = data.get('image')  # Base64 string

        if not (predicted_class and confidence and image_data):
            return jsonify({"error": "Incomplete data"}), 400

        # Decode and save image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        filename = f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        # Save to DB
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)''',
                (session['user_id'], filename, predicted_class, float(confidence)))
            conn.commit()

        return jsonify({"success": True, "filename": filename})

    except Exception as e:
        logger.error(f"Error saving detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_predictions', methods=['POST'])
@login_required
def clear_predictions():
    """Clear user's prediction history"""
    try:
        data = request.get_json()
        clear_type = data.get('type', 'all')  # 'session', 'database', or 'all'
        
        if clear_type in ['session', 'all']:
            # Clear session predictions
            session['predictions'] = []
            session.modified = True
        
        if clear_type in ['database', 'all']:
            # Clear database predictions
            with sqlite3.connect('plant_app.db') as conn:
                c = conn.cursor()
                c.execute('DELETE FROM predictions WHERE user_id = ?', (session['user_id'],))
                conn.commit()
        
        return jsonify({
            'success': True, 
            'message': f'Prediction history cleared successfully',
            'cleared': clear_type
        })
        
    except Exception as e:
        logger.error(f"Clear predictions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_recent', methods=['POST'])
@login_required  
def clear_recent():
    """Clear only recent predictions (last 24 hours)"""
    try:
        # Clear recent session predictions
        session['predictions'] = []
        session.modified = True
        
        # Clear recent database predictions (last 24 hours)
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute('''
                DELETE FROM predictions 
                WHERE user_id = ? 
                AND created_at >= datetime('now', '-1 day')
            ''', (session['user_id'],))
            conn.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Recent predictions cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Clear recent predictions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
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

# Initialize model on startup
def initialize_app():
    """Initialize the application"""
    logger.info("Initializing application...")
    init_db()
    diagnose_model_issue()
    load_model_safely()
    logger.info("Application initialization complete")

initialize_app()

# Application startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Labels loaded: {len(label_dict)} classes")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)