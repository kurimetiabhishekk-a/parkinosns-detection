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

# ── MongoDB (users only: login / register / forgot-password) ──────────────────
MONGODB_URI = (os.environ.get('MONGODB_URI') or '').strip()
if MONGODB_URI.startswith('"') and MONGODB_URI.endswith('"'):
    MONGODB_URI = MONGODB_URI[1:-1]
MONGODB_URI = MONGODB_URI.strip()

if not MONGODB_URI:
    raise ValueError("CRITICAL SECURITY ERROR: MONGODB_URI environment variable is missing. Application cannot start.")

_mongo_client = None   # MongoClient (kept alive for connection pooling)
_mongo_db     = None   # parkisense database handle

def get_users_collection():
    """Return MongoDB 'users' collection. Reconnects automatically if needed."""
    global _mongo_client, _mongo_db

    # --- Use existing connection if available ---
    if _mongo_client is not None:
        try:
            return _mongo_db.users
        except Exception:
            # Handle potential dropped connection handles
            _mongo_client = None
            _mongo_db     = None

    if not MONGODB_URI:
        print("CRITICAL: MONGODB_URI is empty!")
        return None

    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True
        )
        # We'll try to get the database handle. If auth fails later during a query, Flask will catch it.
        _mongo_db = _mongo_client['parkisense']
        # We try to create index but don't strictly crash here
        try:
            _mongo_db.users.create_index('email', unique=True)
        except:
            pass
        return _mongo_db.users
    except Exception as e:
        print(f"CRITICAL: MongoClient setup failed — {e}")
        _mongo_client = None
        _mongo_db     = None
        return None

def init_db():
    """Seed a demo account on startup."""
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('upload', exist_ok=True)
    try:
        col = get_users_collection()
        if col is not None:
            if not col.find_one({'email': 'demo@parkisense.com'}):
                col.insert_one({
                    'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    'name': encrypt_data('Demo User'),
                    'email': 'demo@parkisense.com',
                    'password': generate_password_hash('demo1234'),
                    'pet': encrypt_data('buddy')
                })
                print("DEBUG: Demo user seeded in MongoDB.")
        else:
            print("WARNING: No MongoDB URI set or connection failed. Login will fail until MONGODB_URI is configured.")
    except Exception as e:
        print(f"DEBUG: DB Init Error: {e}")

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
            col = get_users_collection()
            if col is None:
                error = "Database unavailable. Please try again later."
            else:
                user = col.find_one({'email': email})
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
            col = get_users_collection()
            if col is None:
                error = "Database unavailable. Please try again later."
            elif col.find_one({'email': email}):
                error = 'User already registered!'
            else:
                col.insert_one({
                    'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    'name': encrypt_data(name),
                    'email': email,
                    'password': generate_password_hash(password),
                    'pet': encrypt_data(pet)
                })
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
            col = get_users_collection()
            if col is None:
                error = "Database unavailable. Please try again later."
            else:
                # We can't query directly by encrypted pet name unless we use a blind index
                # So we find by email and then verify the pet name
                user = col.find_one({'email': email})
                if user and decrypt_data(user['pet']) == pet:
                    # In a real app, send a reset link. Here we just confirm identity for now.
                    # Returning the password even if hashed isn't useful for the user.
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
    try:
        col = get_users_collection()
        if col is not None:
            # Simple ping
            _mongo_client.admin.command('ping')
            status["mongodb"] = "CONNECTED"
            status["users_count"] = col.count_documents({})
        else:
            status["mongodb"] = "NOT_CONFIGURED (URI empty)"
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
