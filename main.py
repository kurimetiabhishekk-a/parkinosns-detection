# ── Imports ───────────────────────────────────────────────────────────────────
from flask import Flask, render_template, redirect, url_for, request, session
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import os
import time
from utils import *
from voiceTest import *

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'parkisense_secret_2024')
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# ── MongoDB (users only: login / register / forgot-password) ──────────────────
MONGODB_URI = os.environ.get('MONGODB_URI', '')
_mongo_client = None   # MongoClient (kept alive for connection pooling)
_mongo_db     = None   # parkisense database handle

def get_users_collection():
    """Return MongoDB 'users' collection. Reconnects automatically if needed."""
    global _mongo_client, _mongo_db

    # --- health-check cached connection ---
    if _mongo_client is not None:
        try:
            _mongo_client.admin.command('ping', check=False)
            return _mongo_db.users
        except Exception:
            # Connection dropped — reset and fall through to reconnect
            print("DEBUG: MongoDB ping failed, reconnecting...")
            _mongo_client = None
            _mongo_db     = None

    if not MONGODB_URI:
        return None

    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            socketTimeoutMS=8000,
        )
        _mongo_client.admin.command('ping')
        _mongo_db = _mongo_client['parkisense']
        _mongo_db.users.create_index('email', unique=True)
        print("DEBUG: MongoDB connected successfully.")
        return _mongo_db.users
    except Exception as e:
        print(f"CRITICAL: MongoDB connection failed — {e}")
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
                    'name': 'Demo User',
                    'email': 'demo@parkisense.com',
                    'password': 'demo1234',
                    'pet': 'buddy'
                })
                print("DEBUG: Demo user seeded in MongoDB.")
        else:
            print("WARNING: No MongoDB URI set. Login will fail until MONGODB_URI is configured.")
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
                user = col.find_one({'email': email, 'password': password})
                if user:
                    session['name'] = user['name']
                    return redirect(url_for('home'))
                else:
                    error = "Account not found. If new, click 'Create one' to register. (Or use demo@parkisense.com / demo1234)"
        except Exception as e:
            print(f"DEBUG: Login error: {e}")
            error = "Temporary connection issue. Please try again."
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
                    'name': name, 'email': email,
                    'password': password, 'pet': pet
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
                user = col.find_one({'email': email, 'pet': pet})
                if user:
                    error = 'Your password: ' + user['password']
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
        final = 'Weak Pattern'
    elif pred == 'Healthy' and voicePred == 'Healthy':
        final = 'Healthy'
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
            if f and f.filename:
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


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
