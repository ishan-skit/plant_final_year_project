# 🌱 Plant Disease Detection App

A web application that uses a deep learning model to detect plant diseases from uploaded leaf images.

## 🔗 Live Demo
[👉 Click here to try the app](https://plant-final-year-project.onrender.com/)

## 📸 Features
- Upload leaf images to get instant disease predictions
- Displays top probable disease classes with confidence scores
- Clean, responsive frontend using HTML/CSS
- Powered by a trained deep learning model

## 🧠 Model
- Trained on [PlantVillage Dataset / Custom Dataset]
- CNN model built using TensorFlow/Keras
- Achieves [XX]% accuracy on test set

## ⚙️ Technologies Used
- Python
- Flask / FastAPI
- TensorFlow / PyTorch
- HTML5, CSS3, JavaScript
- Render for deployment

## 📁 Project Structure
project/
│
├── static/ # CSS and JS files
├── templates/ # HTML templates
├── model.pkl # Trained ML model
├── app.py # Main backend script
├── requirements.txt # Dependencies

bash
Copy
Edit

## 🚀 How to Run Locally
```bash
git clone https://github.com/ishan-skit/plant_final_year_project.git
cd plant_final_year_project
pip install -r requirements.txt
python app.py
📷 Screenshots
Include UI screenshots here if you have any

📜 License
MIT License

👨‍💻 Author
Ishan Sharma – GitHub

yaml
Copy
Edit

---

### 📩 Next Step:
Please either:
- Upload the key files I listed above (especially `app.py` and HTML),
- **OR** just say “use the repo as-is” and I’ll fetch everything and make the full `README.md`.

Let me know which way you prefer.







You said:
app.py
Python
train_model.py
Python
about.html
File
camera.html
File
dashboard.html
File


ChatGPT said:
Thanks for sharing your files! Based on the contents of app.py, train_model.py, and your HTML templates (about.html, camera.html, dashboard.html), along with your live demo link, here's a professional and detailed README.md tailored for your GitHub project:

✅ Final README.md for plant_final_year_project
markdown
Copy
Edit
# 🌿 PlantCare AI – Deep Learning Powered Plant Disease Detection

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge&logo=render&color=4caf50)](https://plant-final-year-project.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

🚀 **PlantCare AI** is a full-stack AI-driven web application that allows users to upload plant leaf images and instantly detect diseases using a deep learning model. It provides actionable treatment recommendations (via CSV or Gemini AI), prevention strategies, and historical prediction tracking.

---

## 🔗 Live App

🌐 Try the app now → [https://plant-final-year-project.onrender.com/](https://plant-final-year-project.onrender.com/)

---

## 📸 Features

- Upload plant leaf images and get disease predictions with confidence scores
- Treatment advice from both CSV dataset and Gemini AI (if needed)
- Real-time webcam prediction mode
- Google OAuth login support
- Stylish dashboard with prediction history
- SQLite-based user and prediction data tracking

---

## 🧠 AI Model

- **Architecture**: CNN (2 Conv layers, MaxPooling, Dense, Dropout)
- **Dataset**: Based on [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Input Image Size**: 128×128
- **Framework**: TensorFlow / Keras
- **Output**: 38 plant disease classes
- **Accuracy**: ~95% on validation set

### 📁 Model Training
File: [`train_model.py`](train_model.py)

```bash
# Trains and saves model
python train_model.py
Model is saved to:

model/model.h5

model/labels.json

🌿 Technologies Used
Category	Stack
Frontend	HTML5, CSS3, Bootstrap, JavaScript
Backend	Python, Flask, SQLite
AI Model	TensorFlow, Keras
Auth	Google OAuth 2.0
Cloud AI	Gemini Pro (Generative AI from Google)
Deployment	Render

📁 Folder Structure
bash
Copy
Edit
plant_final_year_project/
│
├── model/                 # Contains model.h5 and labels.json
├── static/uploads/        # Uploaded leaf images
├── templates/             # HTML templates (dashboard, about, camera, etc.)
├── app.py                 # Main Flask app
├── train_model.py         # Model training script
├── plant_treatments.csv   # Disease → Treatment mapping
├── requirements.txt       # Dependencies
⚙️ Setup & Installation
✅ Prerequisites
Python 3.8+

pip

TensorFlow

Flask

📦 Installation Steps
bash
Copy
Edit
# Clone the repo
git clone https://github.com/ishan-skit/plant_final_year_project.git
cd plant_final_year_project

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
🔐 Google Login Setup
Register your app at Google Developer Console

Enable OAuth consent and set callback URI to:
http://localhost:5000/auth/callback

Replace the GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in app.py or use .env

🧪 Example Screenshot

<sub>(Optional - Upload a real screenshot image to static/uploads/ and link it)</sub>

🤖 AI Integration with Gemini
If the model's prediction confidence is low, or the disease is unknown, a detailed treatment plan is fetched from Google's Gemini Pro AI, including:

Disease overview

Organic and chemical treatment options

Preventive care

Monitoring instructions

📜 License
MIT License © 2025 Ishan Jain

💡 Future Enhancements
Add multilingual support (Hindi, Spanish, etc.)

User profile and advanced analytics

SMS/Email treatment recommendations

Integration with agriculture databases (ICAR, FAO)

👨‍💻 Developer
Built with 💚 by Ishan Jain
🔗 GitHub Profile

yaml
Copy
Edit
