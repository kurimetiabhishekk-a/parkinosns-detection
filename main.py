# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request, session, Response
from werkzeug.utils import secure_filename
from functools import wraps
import pandas as pd
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
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# ── MongoDB Connection ────────────────────────────────────────────────────────
from pymongo import MongoClient

MONGODB_URI = os.environ.get('MONGODB_URI', '')

def get_db():
    """Return MongoDB database instance."""
    client = MongoClient(MONGODB_URI)
    return client['parkisense']

def init_db():
    """Initialize DB and ensure runtime folders exist."""
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('upload', exist_ok=True)
    try:
        db = get_db()
        # Create indexes for fast lookups
        db.users.create_index('email', unique=True)
        print("DEBUG: MongoDB connected and initialised.")
    except Exception as e:
        print(f"DEBUG: MongoDB init error: {e}")

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
            db = get_db()
            user = db.users.find_one({'email': email, 'password': password})
            if user:
                session['name'] = user['name']
                session['pred'] = 'Healthy'
                session['voicePred'] = 'Healthy'
                return redirect(url_for('home'))
            else:
                error = "Invalid Credentials. Please try again."
        except Exception as e:
            print(f"DEBUG: Login DB error: {e}")
            error = "Database error. Please try again."
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
            db = get_db()
            if db.users.find_one({'email': email}):
                error = 'User already registered!'
                return render_template('register.html', error=error)

            db.users.insert_one({
                'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                'name': name,
                'email': email,
                'password': password,
                'pet': pet
            })
            return redirect(url_for('login'))
        except Exception as e:
            print(f"DEBUG: Register DB error: {e}")
            error = "Registration failed. Please try again."
            return render_template('register.html', error=error)
    return render_template('register.html')


# ── Forgot Password ───────────────────────────────────────────────────────────
@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    error = None
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        pet = (request.form.get('pet') or '').strip().lower()
        try:
            db = get_db()
            user = db.users.find_one({'email': email, 'pet': pet})
            if user:
                error = 'Your password: ' + user['password']
            else:
                error = 'Invalid information. Please try again.'
        except Exception as e:
            print(f"DEBUG: Forgot DB error: {e}")
            error = "Database error. Please try again."
        return render_template('forgot-password.html', error=error)
    return render_template('forgot-password.html')


# ── Home ──────────────────────────────────────────────────────────────────────
@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    return render_template('home.html')


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    user_name = session.get('name', '')
    pred = session.get('pred', 'Healthy')
    voicePred = session.get('voicePred', 'Healthy')

    if pred == 'Parkinson' and voicePred == 'Parkinson':
        final = 'Weak Pattern'
    elif pred == 'Healthy' and voicePred == 'Healthy':
        final = 'Healthy'
    else:
        final = 'Further Diagnosis is Required'

    now = datetime.now()
    try:
        db = get_db()
        db.predictions.insert_one({
            'date': now.strftime("%d/%m/%Y %H:%M:%S"),
            'name': user_name,
            'drawing': 'Weak Pattern' if pred == 'Parkinson' else pred,
            'voice': 'Weak Pattern' if voicePred == 'Parkinson' else voicePred,
            'final': final
        })
        records = list(db.predictions.find({'name': user_name}, {'_id': 0}))
        df = pd.DataFrame(records)
        if not df.empty:
            df.columns = ['Date', 'Name', 'DrawingPrediction', 'VoicePrediction', 'FinalPrediction']
    except Exception as e:
        print(f"DEBUG: Dashboard DB error: {e}")
        df = pd.DataFrame(columns=['Date', 'Name', 'DrawingPrediction', 'VoicePrediction', 'FinalPrediction'])

    now_date = now.strftime("%d %B %Y, %H:%M")
    return render_template('dashboard.html',
                           tables=[df.to_html(classes='table-responsive table table-bordered table-hover')],
                           titles=df.columns.values, now_date=now_date)


# ── Image / Spiral Test ───────────────────────────────────────────────────────
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
                print(f"DEBUG: Error saving canvas drawing: {e}")
        f = request.files.get('doc')
        if f:
            f.save(os.path.join(savepath, secure_filename('test.jpg')))
            return redirect(url_for('image_test'))
    return render_template('image.html')


@app.route('/image_test', methods=['GET', 'POST'])
@login_required
def image_test():
    label, result, suggestion, accuracy = predictImg(r'static/img/test.jpg')
    if label is not None:
        session['pred'] = label
    return render_template('image_test.html', result=result, suggestion=suggestion, confidence=accuracy)


# ── Voice Test ────────────────────────────────────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if request.form.get('uploadbutton') == 'Upload':
            savepath = 'upload/'
            os.makedirs(savepath, exist_ok=True)
            f = request.files.get('doc')
            if f:
                f.save(os.path.join(savepath, secure_filename('test.wav')))
                return render_template('upload.html', file=f.filename, mgs='File uploaded..!!')
        elif request.form.get('uploadbutton') == 'Detect PD':
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
