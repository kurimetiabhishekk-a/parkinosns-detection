import os
import time
import base64
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, redirect, url_for, request, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet
from dotenv import load_dotenv

from utils import *
from voiceTest import *

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = (os.environ.get('SECRET_KEY') or 'parkisense_secure_default_key').strip()

# ── Encryption Setup ──────────────────────────────────────────────────────────
ENCRYPTION_KEY = (os.environ.get('ENCRYPTION_KEY') or '').strip()
fernet = None
if ENCRYPTION_KEY:
    try:
        fernet = Fernet(ENCRYPTION_KEY.encode())
    except Exception as e:
        print(f"ERROR: Invalid ENCRYPTION_KEY format: {e}")

def encrypt_data(data):
    """Encrypt a string if fernet is available."""
    if not fernet or not data:
        return data
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(data):
    """Decrypt a string if fernet is available."""
    if not fernet or not data:
        return data
    try:
        return fernet.decrypt(data.encode()).decode()
    except Exception:
        return "[Encrypted]"
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True

import sqlite3

def get_db_connection():
    conn = sqlite3.connect('parkisense.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('upload', exist_ok=True)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                name TEXT,
                password TEXT,
                pet TEXT,
                date TEXT
            )
        ''')
        # Seed demo user
        c.execute("SELECT * FROM users WHERE email = 'demo@parkisense.com'")
        if not c.fetchone():
            c.execute("INSERT INTO users (email, name, password, pet, date) VALUES (?, ?, ?, ?, ?)",
                      ('demo@parkisense.com', encrypt_data('Demo User'), generate_password_hash('demo1234'), encrypt_data('buddy'), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        conn.commit()
        conn.close()
        print("DEBUG: SQLite DB initialized and demo user seeded.")
    except Exception as e:
        print(f"DEBUG: SQLite Init Error: {e}")

init_db()


@app.context_processor
def inject_now():
    return {
        'now': datetime.now(),
        'timestamp': int(time.time()),
        'name': session.get('name', '')
    }


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'name' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def landing():
    return redirect(url_for('login'))


# ── Login ─────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            conn.close()
            
            if user and check_password_hash(user['password'], password):
                session['name'] = decrypt_data(user['name'])
                return redirect(url_for('home'))
            else:
                error = "Invalid email or password."
        except Exception as e:
            print(f"DEBUG: Login error: {e}")
            error = "Database unreachable. Please try again."
        
        return render_template('login.html', error=error)
    return render_template('login.html')


# ── Logout ────────────────────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── Register ──────────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST' and request.form.get('sub') == 'Submit':
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        rpassword = request.form.get('rpassword') or ''
        pet = (request.form.get('pet') or '').strip().lower()

        if password != rpassword:
            error = 'Passwords do not match!'
            return render_template('register.html', error=error)

        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT email FROM users WHERE email = ?", (email,))
            if c.fetchone():
                error = 'User already registered!'
                conn.close()
            else:
                c.execute("INSERT INTO users (email, name, password, pet, date) VALUES (?, ?, ?, ?, ?)",
                          (email, encrypt_data(name), generate_password_hash(password), encrypt_data(pet), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
        except Exception as e:
            print(f"DEBUG: Register error: {e}")
            error = "Database busy. Please try again."
    return render_template('register.html', error=error)


# ── Forgot Password ───────────────────────────────────────────────────────────
@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    error = None
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        pet = (request.form.get('pet') or '').strip().lower()
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            conn.close()
            
            if user and hasattr(user, 'keys') and 'pet' in user.keys() and decrypt_data(user['pet']) == pet:
                error = 'Identity verified. Please contact admin to reset password (feature coming soon).'
            elif user and not hasattr(user, 'keys') and decrypt_data(user[4]) == pet: # SQLite Row index 4 is pet
                 error = 'Identity verified. Please contact admin to reset password (feature coming soon).'
            else:
                error = 'Information not found.'
        except Exception as e:
            print(f"DEBUG: Forgot error: {e}")
            error = "Database busy. Please try again."
        return render_template('forgot-password.html', error=error)
    return render_template('forgot-password.html')


# ── Home ──────────────────────────────────────────────────────────────────────
@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    return render_template('home.html')


# ── Dashboard ───────────────────────────────────────────────────────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    pred      = session.get('pred', 'Healthy')
    voicePred = session.get('voicePred', 'Healthy')

    if pred == 'Parkinson' and voicePred == 'Parkinson':
        final = 'Clinical Consultation Recommended'
    elif pred == 'Healthy' and voicePred == 'Healthy':
        final = 'Healthy (Low Risk)'
    elif pred == 'Parkinson' or voicePred == 'Parkinson':
        final = 'Potential Early Indicators'
    else:
        final = 'Further Diagnosis Required'

    now_date = datetime.now().strftime("%d %B %Y, %H:%M")
    return render_template('dashboard.html',
                           pred=pred, voice_pred=voicePred,
                           final=final, now_date=now_date)


# ── Spiral/Image Test ─────────────────────────────────────────────────────────
@app.route('/image', methods=['GET', 'POST'])
@login_required
def image():
    if request.method == 'POST':
        savepath = 'static/img/'
        os.makedirs(savepath, exist_ok=True)
        drawing_data = request.form.get('drawing_data')
        if drawing_data and ',' in drawing_data:
            import base64
            try:
                header, encoded = drawing_data.split(",", 1)
                data = base64.b64decode(encoded)
                with open(os.path.join(savepath, 'test.jpg'), 'wb') as f:
                    f.write(data)
                return redirect(url_for('image_test'))
            except Exception as e:
                print(f"DEBUG: Canvas save error: {e}")
        f = request.files.get('doc')
        if f and f.filename:
            f.save(os.path.join(savepath, 'test.jpg'))
            return redirect(url_for('image_test'))
    return render_template('image.html')


@app.route('/image_test', methods=['GET', 'POST'])
@login_required
def image_test():
    label, result, suggestion, accuracy = predictImg(r'static/img/test.jpg')
    if label is not None:
        session['pred'] = label
    return render_template('image_test.html', result=result, suggestion=suggestion, confidence=accuracy)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        savepath = 'upload/'
        os.makedirs(savepath, exist_ok=True)
        f = request.files.get('doc')
        
        if request.form.get('uploadbutton') == 'Upload':
            if f and f.filename:
                f.save(os.path.join(savepath, 'test.wav'))
                return render_template('upload.html', file=f.filename, mgs='File uploaded..!!')
        elif request.form.get('uploadbutton') == 'Detect PD':
            audio_b64 = request.form.get('audio_base64')
            if audio_b64 and ',' in audio_b64:
                import base64
                header, encoded = audio_b64.split(",", 1)
                try:
                    with open(os.path.join(savepath, 'test.wav'), 'wb') as f_out:
                        f_out.write(base64.b64decode(encoded))
                except Exception as e:
                    print(f"DEBUG: Base64 decode error: {e}")
            elif f and f.filename:
                f.save(os.path.join(savepath, 'test.wav'))
                
            voice_result = testVoice()
            if len(voice_result) == 2:
                _, msg = voice_result
                return render_template('upload.html', mgs=msg, accuracy=None)
            label, result, accuracy = voice_result
            session['voicePred'] = label
            return render_template('upload.html', mgs=result, accuracy=accuracy)
    return render_template('upload.html')


@app.route('/record', methods=['GET', 'POST'])
@login_required
def record():
    return render_template('record.html')


# ── Diagnostic Route ──────────────────────────────────────────────────────────
@app.route('/diagnose')
def diagnose():
    """Check system status for troubleshooting."""
    status = {
        "status": "online",
        "mongodb": "unknown",
        "encryption": "unknown",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Test MongoDB
    # Test SQLite DB
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        count = c.fetchone()[0]
        status["mongodb"] = "CONNECTED (SQLite DB)"
        status["users_count"] = count
        conn.close()
    except Exception as e:
        status["mongodb"] = f"ERROR: {str(e)}"

    # Test Encryption
    if fernet:
        status["encryption"] = "READY"
    else:
        status["encryption"] = "NOT_READY (Invalid format or key missing)"

    return status


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
