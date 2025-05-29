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

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# App Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application")

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///plant_app.db')
db = SQLAlchemy(app)

# Configure uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Google OAuth Configuration
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

# Check required environment variables
required_vars = ['GEMINI_API_KEY', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing_vars)}")

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MODEL_PATH = 'model/model.h5'
LABELS_PATH = 'model/labels.json'
TREATMENTS_PATH = 'plant_treatments.csv'

# Global variables
model = None
label_dict = {}
treatments_df = pd.DataFrame()
ai_cache = {}  # Cache for AI responses
ai_cache_lock = Lock()

# Load model and labels
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    label_dict = {v: k for k, v in label_map.items()}
    print("Model and labels loaded successfully")
except Exception as e:
    model = None
    label_dict = {}
    print(f"Error loading model or labels: {e}")

# Load treatment data
try:
    treatments_df = pd.read_csv(TREATMENTS_PATH)
    print("Treatment data loaded successfully")
except Exception as e:
    treatments_df = pd.DataFrame()
    print(f"Error loading treatment data: {e}")

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

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
        
        # Check if google_id column exists, if not add it
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        if 'google_id' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
            print("Added google_id column to users table")
        
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

init_db()

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Optimized image preprocessing"""
    try:
        # Use PIL for faster loading and resizing
        img = Image.open(img_path)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize((128, 128), Image.Resampling.LANCZOS)  # Faster resizing
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def predict_disease(img_path):
    """Optimized disease prediction"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        print(f"Starting prediction for: {img_path}")
        start_time = time.time()
        
        img = preprocess_image(img_path)
        print(f"Image preprocessing took: {time.time() - start_time:.2f}s")
        
        pred_start = time.time()
        preds = model.predict(img, verbose=0)  # verbose=0 for faster prediction
        print(f"Model prediction took: {time.time() - pred_start:.2f}s")
        
        class_index = np.argmax(preds)
        class_name = label_dict.get(class_index, "Unknown")
        confidence = float(np.max(preds))
        
        print(f"Total prediction time: {time.time() - start_time:.2f}s")
        return class_name, confidence
        
    except Exception as e:
        print(f"Error in predict_disease: {e}")
        return "Prediction Error", 0.0

def get_ai_treatment_async(disease_name, confidence_score=0.0, timeout=15):
    """Get treatment from Gemini AI with timeout and caching"""
    
    # Create cache key
    cache_key = f"{disease_name}_{confidence_score:.2f}"
    
    # Check cache first
    with ai_cache_lock:
        if cache_key in ai_cache:
            print(f"Using cached AI response for: {disease_name}")
            return ai_cache[cache_key]
    
    def get_ai_response():
        try:
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
            
            # Shorter, more focused prompt for faster response
            prompt = f"""
            Plant disease: "{disease_name}"
            Confidence: {confidence_score:.1%}
            
            Provide concise treatment info:
            1. Treatment: Main treatment method
            2. Prevention: Key prevention steps
            3. Severity: Mild/Moderate/Severe
            
            Keep response under 300 words.
            """
            
            response = model_ai.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,  # Limit response length
                    temperature=0.1,  # More focused responses
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
                    'source': 'Gemini AI-generated'
                }
                
                # Cache the result
                with ai_cache_lock:
                    ai_cache[cache_key] = result
                
                return result
            else:
                raise Exception("No response from Gemini AI")
                
        except Exception as e:
            print(f"Gemini AI error: {e}")
            raise e
    
    # Use ThreadPoolExecutor with timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_ai_response)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"AI request timed out after {timeout}s for disease: {disease_name}")
        return get_fallback_treatment(disease_name)
    except Exception as e:
        print(f"AI request failed: {e}")
        return get_fallback_treatment(disease_name)

def get_fallback_treatment(disease_name):
    """Fallback treatment when AI fails"""
    return {
        'disease': disease_name,
        'treatment': 'Unable to generate AI treatment at this time. Please consult with a plant specialist for proper diagnosis and treatment.',
        'prevention': 'Follow general plant care: proper watering, good drainage, adequate spacing, and regular monitoring.',
        'organic_treatment': 'Apply organic compost, neem oil, or copper-based fungicides as appropriate.',
        'chemical_treatment': 'Consult agricultural expert for chemical treatments if organic methods fail.',
        'severity': 'Unknown - requires professional assessment',
        'source': 'Fallback'
    }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment information - prioritize CSV, fallback to fast AI"""
    
    CONFIDENCE_THRESHOLD = 0.7
    
    # First, try CSV database
    if not treatments_df.empty and confidence_score >= CONFIDENCE_THRESHOLD:
        treatment_row = treatments_df[treatments_df['disease'] == disease_name]
        
        if treatment_row.empty:
            treatment_row = treatments_df[treatments_df['disease'].str.contains(disease_name, case=False, na=False)]
        
        if not treatment_row.empty:
            row = treatment_row.iloc[0]
            print(f"Using CSV treatment for: {disease_name}")
            return {
                'disease': row['disease'],
                'treatment': row['treatment'],
                'prevention': row['prevention'],
                'organic_treatment': row['organic_treatment'],
                'chemical_treatment': row['chemical_treatment'],
                'severity': row['severity'],
                'source': 'Database'
            }
    
    # Fallback to AI (with timeout and caching)
    print(f"Using AI treatment for: {disease_name} (confidence: {confidence_score:.1%})")
    return get_ai_treatment_async(disease_name, confidence_score, timeout=10)

def getImmediateAction(disease_name, confidence=0.0):
    """Wrapper for get_treatment_info to maintain template compatibility"""
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
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not username or not email or not password:
            flash("All fields are required!")
            return redirect(request.url)

        hashed_pw = generate_password_hash(password)
        try:
            with sqlite3.connect('plant_app.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                          (username, email, hashed_pw))
                conn.commit()
            flash("Registration successful! Please log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.")
    return render_template('login.html')

@app.route('/auth/google')
def google_login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def google_callback():
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
        print(f"OAuth error: {e}")
        flash("Google login failed. Please try again.")
        return redirect(url_for('login'))
    
    flash("Google login failed.")
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
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print("Starting prediction process...")
            predicted_class, confidence = predict_disease(filepath)
            print(f"Prediction complete: {predicted_class} ({confidence:.2%})")
            
            print("Getting treatment info...")
            treatment_info = get_treatment_info(predicted_class, confidence)
            print("Treatment info retrieved")

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
        flash("Invalid file format or no file selected.")
    return render_template('predict.html')

@app.route('/dashboard')
@login_required
@cache.cached(timeout=60)
def dashboard():
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
                        getImmediateAction=getImmediateAction)

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

@app.route('/camera')
@login_required
def camera():
    """Real-time camera detection page"""
    return render_template('camera.html')

@app.route('/api/process_frame', methods=['POST'])
@login_required
def process_frame():
    """Process individual camera frame for disease detection"""
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
            
            # Only get treatment info for high-confidence predictions in real-time
            treatment_info = None
            if confidence > 0.5:
                treatment_info = get_treatment_info(predicted_class, confidence)
            
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'treatment_info': treatment_info
            })
            
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise e
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_detection', methods=['POST'])
@login_required
def save_detection():
    """Save a detection result from real-time camera"""
    try:
        data = request.get_json()
        
        if not all(key in data for key in ['image', 'predicted_class', 'confidence']):
            return jsonify({'error': 'Missing required data'}), 400
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"camera_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        with sqlite3.connect('plant_app.db') as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)''',
                (session['user_id'], filename, data['predicted_class'], data['confidence'] / 100))
            conn.commit()
            prediction_id = c.lastrowid
        
        return jsonify({
            'success': True,
            'message': 'Detection saved successfully!',
            'prediction_id': prediction_id
        })
        
    except Exception as e:
        print(f"Error saving detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_camera', methods=['POST'])
@login_required  
def predict_camera():
    """Handle camera capture from regular predict page"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        filename = f"capture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        predicted_class, confidence = predict_disease(filepath)
        treatment_info = get_treatment_info(predicted_class, confidence)
        
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
        print(f"Error processing camera capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_treatment', methods=['POST'])
@login_required
def api_ai_treatment():
    data = request.get_json()
    disease = data.get('disease')
    confidence = data.get('confidence', 0.0)
    
    if not disease:
        return jsonify({"error": "Disease name is required"}), 400
    
    treatment_info = get_ai_treatment_async(disease, confidence, timeout=15)
    return jsonify(treatment_info)

@app.route('/test_gemini')
@login_required
def test_gemini():
    try:
        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        response = model_ai.generate_content("Hello, please respond with 'Gemini API is working correctly!'")
        return jsonify({
            "status": "success",
            "message": response.text if response and response.text else "No response"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)