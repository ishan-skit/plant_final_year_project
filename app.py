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

# Load model and labels
try:
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
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_disease(img_path):
    if model is None:
        return "Model not loaded", 0.0
    img = preprocess_image(img_path)
    preds = model.predict(img)
    class_index = np.argmax(preds)
    class_name = label_dict.get(class_index, "Unknown")
    confidence = float(np.max(preds))
    return class_name, confidence

def get_ai_treatment(disease_name, confidence_score=0.0):
    """Get treatment information from Gemini AI when disease is unknown or confidence is low"""
    try:
        model_ai = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        As a plant pathology expert, provide comprehensive treatment information for the plant disease: "{disease_name}".

        Please provide a structured response with the following sections:
        1. Disease Description: Brief overview of the disease
        2. Treatment: Detailed treatment protocol including chemical and organic options
        3. Prevention: Preventive measures to avoid future infections
        4. Severity Assessment: Rate the severity (Mild/Moderate/Severe)
        5. Monitoring: What to watch for during treatment

        If the disease name seems unclear or generic, provide general plant health advice.
        Keep the response practical and actionable for home gardeners and farmers.
        Format your response in a clear, organized manner.
        """

        response = model_ai.generate_content(prompt)
        
        if response and response.text:
            ai_treatment = response.text.strip()
            return {
                'disease': disease_name,
                'treatment': ai_treatment,
                'prevention': 'AI-generated prevention advice included above',
                'organic_treatment': 'AI-generated organic treatment included above',
                'chemical_treatment': 'AI-generated chemical treatment included above',
                'severity': 'As assessed by AI',
                'source': 'Gemini AI-generated'
            }
        else:
            raise Exception("No response from Gemini AI")
        
    except Exception as e:
        print(f"Gemini AI treatment generation error: {e}")
        return {
            'disease': disease_name,
            'treatment': f'AI treatment generation failed: {str(e)}. Please consult with a plant specialist for proper diagnosis and treatment.',
            'prevention': 'Follow general plant care guidelines including proper watering, spacing, and monitoring.',
            'organic_treatment': 'Apply organic compost and maintain good plant hygiene.',
            'chemical_treatment': 'Consult with agricultural expert for appropriate chemical treatments.',
            'severity': 'Unknown',
            'source': 'Fallback'
        }

def get_treatment_info(disease_name, confidence_score=0.0):
    """Get treatment information for a disease from CSV or AI"""
    
    CONFIDENCE_THRESHOLD = 0.7
    
    use_ai = (confidence_score < CONFIDENCE_THRESHOLD or 
              disease_name.lower() in ['unknown', 'unidentified', 'unclear'] or
              treatments_df.empty)
    
    if not use_ai:
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
    
    print(f"Using Gemini AI for treatment - Disease: {disease_name}, Confidence: {confidence_score}")
    return get_ai_treatment(disease_name, confidence_score)

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
            
            treatment_info = None
            if confidence > 0.3:
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
    
    treatment_info = get_ai_treatment(disease, confidence)
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