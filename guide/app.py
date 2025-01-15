from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle
import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import torch
from torchvision import transforms
from PIL import Image
import uuid  # For generating unique filenames
from model import load_model  # Only load_model is imported now

app = Flask(__name__)
app.secret_key = 'chin tapak dum dum'

# Load the trained model for plant disease identification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model('model.pth')  # Load your trained ResNet18 model here
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List of class names directly in app.py for plant disease identification
classnames = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight',
    'Potato healthy', 'Potato Late blight', 'Tomato Target Spot', 
    'Tomato Tomato mosaic virus', 'Tomato Yellow Leaf Curl Virus',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato healthy',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 
    'Tomato Spider mites'
]

# Define the correct paths for model and scaler files for crop recommendation
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'GBmodel.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
LE_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

# Load the Gradient Boosting model for crop recommendation
gb_model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))
le = pickle.load(open(LE_PATH, 'rb'))

# MySQL connection configuration
db_config = {
    'user': 'user',
    'password': 'password',
    'host': '127.0.0.1',
    'database': 'login'
}

# Function to connect to MySQL
def create_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            print("Connection to MySQL DB successful")
            return connection
    except Error as e:
        print(f"The error '{e}' occurred")
        return None

def validate_input(N, P, K, temperature, humidity, ph, rainfall):
    if not (0 <= N <= 150 and 5 <= P <= 150 and 5 <= K <= 210):
        return False
    if not (7 <= temperature <= 50):
        return False
    if not (14 <= humidity <= 100):
        return False
    if not (3.5 <= ph <= 10):
        return False
    if not (20 <= rainfall <= 300):
        return False
    return True

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Crop recommendation route
@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('index.html')

# Prediction route for crop recommendation
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        if not validate_input(N, P, K, temperature, humidity, ph, rainfall):
            return render_template('result.html', crop="Invalid input, no crop found")

        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_features_scaled = scaler.transform(input_features)
        prediction = gb_model.predict(input_features_scaled)
        predicted_crop = le.inverse_transform(prediction)

        return render_template('result.html', crop=predicted_crop[0])

# Plant disease identification routes
@app.route('/plant-disease')
def plant_disease():
    return render_template('index2.html')

@app.route('/predictd', methods=['POST'])
def predict_plant_disease():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded file with a unique filename
    filename = f"{uuid.uuid4()}_{file.filename}"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    # Process the image
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        # Debugging: Print the predicted index to check if the model is predicting correctly
        predicted_index = predicted.item()
        print(f"Predicted index: {predicted_index}")
        print(f"Class names list length: {len(classnames)}")  # Print the length of classnames for debugging
        
        # Ensure predicted index is within bounds
        if predicted_index < len(classnames):
            classname = classnames[predicted_index]
            print(f"Predicted class name: {classname}")  # Print the predicted class name for debugging
        else:
            classname = "Unknown class"  # In case something goes wrong with the index
            print("Predicted index out of bounds")

    # Converting the file path to a URL path for rendering in HTML
    img_url_path = f'/static/uploads/{filename}'

    return render_template('result2.html', classname=classname, img_path=img_url_path)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None 
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
            user = cursor.fetchone()
            conn.close()
            if user:
                session['user_id'] = user[0]
                session['name'] = user[1]  
                session['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return redirect(url_for('home'))
            else:
                error_message = "Invalid credentials, please try again" 
    return render_template('login.html', error_message=error_message)


# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error_message = None  # To store any signup errors
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        conn = create_connection()
        if conn:
            cursor = conn.cursor()

            # Check if the email already exists
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                error_message = "Email already registered, please use a different email."
            else:
                # Insert new user if email is not already registered
                cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
    return render_template('signup.html', error_message=error_message)  # Pass error message to template


# Logout route
@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
