# Flask Plant Disease Detection App
import cv2
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from keras.utils import load_img, img_to_array
import tensorflow as tf
from datetime import datetime
from functools import wraps
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
import concurrent.futures
from threading import Lock

# Configure TensorFlow to use CPU only and avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow for CPU-only deployment
tf.config.set_visible_devices([], 'GPU')

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"Gemini API not configured: {e}")
    GEMINI_AVAILABLE = False

# App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Google OAuth Configuration (optional)
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

# Initialize OAuth only if credentials are available
oauth = None
google = None
if app.config['GOOGLE_CLIENT_ID'] and app.config['GOOGLE_CLIENT_SECRET']:
    try:
        oauth = OAuth(app)
        google = oauth.register(
            name='google',
            client_id=app.config['GOOGLE_CLIENT_ID'],
            client_secret=app.config['GOOGLE_CLIENT_SECRET'],
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
        GOOGLE_AUTH_AVAILABLE = True
    except Exception as e:
        print(f"Google OAuth not configured: {e}")
        GOOGLE_AUTH_AVAILABLE = False
else:
    GOOGLE_AUTH_AVAILABLE = False

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'

# Global variables
model = None
label_dict = {}
treatments_df = pd.DataFrame()
ai_cache = {}
ai_cache_lock = Lock()
MODEL_LOADED = False

def load_model_safe():
    """Safely load the model with proper error handling"""
    global model, label_dict, MODEL_LOADED
    
    try:
        print("Loading TensorFlow model...")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at: {MODEL_PATH}")
            return False
            
        if not os.path.exists(LABELS_PATH):
            print(f"❌ Labels file not found at: {LABELS_PATH}")
            return False
        
        # Load model with CPU configuration
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
        # Recompile for CPU
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load labels
        with open(LABELS_PATH, 'r') as f:
            label_map = json.load(f)
        label_dict = {v: k for k, v in label_map.items()}
        
        print(f"✅ Model loaded successfully with {len(label_dict)} classes")
        MODEL_LOADED = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        label_dict = {}
        MODEL_LOADED = False
        return False

def load_treatments_safe():
    """Safely load treatment data"""
    global treatments_df
    try:
        if os.path.exists(TREATMENTS_PATH):
            treatments_df = pd.read_csv(TREATMENTS_PATH)
            print(f"✅ Treatment data loaded: {len(treatments_df)} treatments")
        else:
            print(f"⚠️ Treatment file not found: {TREATMENTS_PATH}")
            treatments_df = pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading treatments: {e}")
        treatments_df = pd.DataFrame()

# Initialize model and data
print("Initializing application...")
load_model_safe()
load_treatments_safe()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": MODEL_LOADED,
        "gemini_available": GEMINI_AVAILABLE,
        "google_auth_available": GOOGLE_AUTH_AVAILABLE
    }), 200

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize SQLite DB
def init_db():
    """Initialize database with proper error handling"""
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
            
            # Check if google_id column exists
            c.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in c.fetchall()]
            if 'google_id' not in columns:
                c.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
                print("✅ Added google_id column to users table")
            
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
            print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

init_db()

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Optimized image preprocessing with error handling"""
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise

def predict_disease(img_path):
    """Disease prediction with fallback handling"""
    if not MODEL_LOADED or model is None:
        return "Model not available", 0.0
    
    try:
        print(f"🔍 Starting prediction for: {img_path}")
        start_time = time.time()
        
        img = preprocess_image(img_path)
        
        # Use CPU for prediction
        with tf.device('/CPU:0'):
            preds = model.predict(img, verbose=0)
        
        class_index = np.argmax(preds)
        class_name = label_dict.get(class_index, "Unknown")
        confidence = float(np.max(preds))
        
        print(f"✅ Prediction complete: {class_name} ({confidence:.2%}) in {time.time() - start_time:.2f}s")
        return class_name, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Prediction Error", 0.0

def get_ai_treatment_async(disease_name, confidence_score=0.0, timeout=10):
    """Get AI treatment with proper fallback"""
    if not GEMINI_AVAILABLE:
        return get_fallback_treatment(disease_name)
    
    cache_key = f"{disease_name}_{confidence_score:.2f}"
    
    with ai_cache_lock:
        if cache_key in ai_cache:
            return ai_cache[cache_key]
    
    def get_ai_response():
        try:
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Plant disease: "{disease_name}"
            Confidence: {confidence_score:.1%}
            
            Provide concise treatment info:
            1. Treatment: Main treatment method
            2. Prevention: Key prevention steps  
            3. Severity: Mild/Moderate/Severe
            
            Keep response under 200 words.
            """
            
            response = model_ai.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1,
                )
            )
            
            if response and response.text:
                result = {
                    'disease': disease_name,
                    'treatment': response.text.strip(),
                    'prevention': 'See AI response above',
                    'organic_treatment': 'Included in main treatment',
                    'chemical_treatment': 'Included in main treatment',
                    'severity': 'As assessed by AI',
                    'source': 'Gemini AI'
                }
                
                with ai_cache_lock:
                    ai_cache[cache_key] = result
                
                return result
            else:
                raise Exception("No response from Gemini AI")
                
        except Exception as e:
            print(f"❌ Gemini AI error: {e}")
            raise e
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_ai_response)
            return future.result(timeout=timeout)
    except (concurrent.futures.TimeoutError, Exception) as e:
        print(f"❌ AI request failed: {e}")
        return get_fallback_treatment(disease_name)

def get_fallback_treatment(disease_name):
    """Enhanced fallback treatment"""
    return {
        'disease': disease_name,
        'treatment': f'For {disease_name}: Remove affected parts, improve air circulation, avoid overhead watering. Consult agricultural expert for specific treatment.',
        'prevention': 'Proper spacing, good drainage, regular monitoring, and appropriate fertilization.',
        'organic_treatment': 'Neem oil, copper fungicide, or baking soda solution may help depending on the condition.',
        'chemical_treatment': 'Consult agricultural extension service for appropriate chemical treatments.',
        'severity': 'Requires assessment - monitor plant closely',
        'source': 'General Guidelines'
    }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment info with CSV priority, AI fallback"""
    
    # Try CSV first for high confidence
    if not treatments_df.empty and confidence_score >= 0.6:
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
    
    # Fallback to AI or general treatment
    return get_ai_treatment_async(disease_name, confidence_score, timeout=8)

def getImmediateAction(disease_name, confidence=0.0):
    """Wrapper for template compatibility"""
    treatment = get_treatment_info(disease_name, confidence)
    return treatment.get('treatment', 'No specific treatment information available')

# Routes
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
            return redirect(request.url)

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
            print(f"Registration error: {e}")
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
            return redirect(request.url)

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
            print(f"Login error: {e}")
            flash("Login failed. Please try again.")
    
    return render_template('login.html', google_auth_available=GOOGLE_AUTH_AVAILABLE)

@app.route('/auth/google')
def google_login():
    if not GOOGLE_AUTH_AVAILABLE:
        flash("Google login is not available.")
        return redirect(url_for('login'))
    
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    try:
        redirect_uri = url_for('google_callback', _external=True)
        return google.authorize_redirect(redirect_uri)
    except Exception as e:
        print(f"Google login error: {e}")
        flash("Google login failed.")
        return redirect(url_for('login'))

@app.route('/auth/callback')
def google_callback():
    if not GOOGLE_AUTH_AVAILABLE:
        flash("Google login is not available.")
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
                    username = name.replace(' ', '_').lower()
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
        print(f"OAuth callback error: {e}")
        flash("Google login failed. Please try again.")
    
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if not MODEL_LOADED:
            flash("Model is not available. Please try again later.")
            return redirect(request.url)
        
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                predicted_class, confidence = predict_disease(filepath)
                treatment_info = get_treatment_info(predicted_class, confidence)

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
                print(f"Prediction error: {e}")
                flash("Error processing image. Please try again.")
        else:
            flash("Invalid file format or no file selected.")
    
    return render_template('predict.html', model_loaded=MODEL_LOADED)

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
                LIMIT 10
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
            stats = c.fetchone()
        
        return render_template('dashboard.html', 
                            predictions=predictions,
                            stats=stats,
                            getImmediateAction=getImmediateAction,
                            model_loaded=MODEL_LOADED)
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash("Error loading dashboard.")
        return render_template('dashboard.html', 
                            predictions=[],
                            stats=(0, 0, 0, 0),
                            getImmediateAction=getImmediateAction,
                            model_loaded=MODEL_LOADED)

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

# API Routes with better error handling
@app.route('/api/process_frame', methods=['POST'])
@login_required
def process_frame():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        temp_filename = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_filepath)
        
        try:
            predicted_class, confidence = predict_disease(temp_filepath)
            
            treatment_info = None
            if confidence > 0.4:
                treatment_info = get_treatment_info(predicted_class, confidence)
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'treatment_info': treatment_info
            })
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
    except Exception as e:
        print(f"Frame processing error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting Plant Disease Detection App...")
    print(f"Model loaded: {MODEL_LOADED}")
    print(f"Gemini AI available: {GEMINI_AVAILABLE}")
    print(f"Google Auth available: {GOOGLE_AUTH_AVAILABLE}")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))