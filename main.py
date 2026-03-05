# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request, session, Response
from werkzeug.utils import secure_filename
from functools import wraps
import pandas as pd
from datetime import datetime
import os
import time
import sqlite3
from utils import *
from voiceTest import *

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'parkisense_secret_2024')
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# ── Resilient Database Logic (MongoDB + SQLite Fallback) ─────────────────────
MONGODB_URI = os.environ.get('MONGODB_URI', '')

def get_mongodb_client():
    if MONGODB_URI:
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Trigger a ping to check connection
            client.admin.command('ping')
            return client
        except Exception as e:
            print(f"CRITICAL: MongoDB Connection Failed: {e}. Falling back to SQLite.")
    return None

def get_db():
    client = get_mongodb_client()
    if client:
        return client['parkisense'], 'mongo'
    else:
        conn = sqlite3.connect('mydatabase.db', check_same_thread=False)
        return conn, 'sqlite'

def init_db():
    """Initialise tables/collections and seed a default user."""
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('upload', exist_ok=True)
    try:
        db_obj, db_type = get_db()
        if db_type == 'mongo':
            # MongoDB Initialisation
            db_obj.users.create_index('email', unique=True)
            if not db_obj.users.find_one({'email': 'demo@parkisense.com'}):
                db_obj.users.insert_one({
                    'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    'name': 'Demo User',
                    'email': 'demo@parkisense.com',
                    'password': 'demo1234',
                    'pet': 'buddy'
                })
            print("DEBUG: MongoDB initialised and seeded.")
        else:
            # SQLite Initialisation
            cur = db_obj.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    date TEXT, name TEXT NOT NULL, email TEXT PRIMARY KEY, 
                    password TEXT NOT NULL, pet TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    date TEXT, name TEXT, drawing_pred TEXT, 
                    voice_pred TEXT, final_pred TEXT
                )
            """)
            cur.execute("SELECT name FROM users WHERE email=?", ("demo@parkisense.com",))
            if not cur.fetchone():
                dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                cur.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?)",
                    (dt, "Demo User", "demo@parkisense.com", "demo1234", "buddy")
                )
            db_obj.commit()
            db_obj.close()
            print("DEBUG: SQLite initialised and seeded (Fallback mode).")
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
            db_obj, db_type = get_db()
            if db_type == 'mongo':
                user = db_obj.users.find_one({'email': email, 'password': password})
                if user:
                    session['name'] = user['name']
                    return redirect(url_for('home'))
            else:
                cur = db_obj.cursor()
                cur.execute("SELECT name FROM users WHERE email=? AND password=?", (email, password))
                row = cur.fetchone()
                db_obj.close()
                if row:
                    session['name'] = row[0]
                    return redirect(url_for('home'))
            
            error = "Invalid Credentials. Use demo@parkisense.com / demo1234 if you're testing."
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
            db_obj, db_type = get_db()
            if db_type == 'mongo':
                if db_obj.users.find_one({'email': email}):
                    error = 'User already registered!'
                else:
                    db_obj.users.insert_one({
                        'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'name': name,
                        'email': email,
                        'password': password,
                        'pet': pet
                    })
                    return redirect(url_for('login'))
            else:
                cur = db_obj.cursor()
                cur.execute("SELECT name FROM users WHERE email=?", (email,))
                if cur.fetchone():
                    error = 'User already registered!'
                else:
                    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    cur.execute("INSERT INTO users VALUES (?,?,?,?,?)", (dt, name, email, password, pet))
                    db_obj.commit()
                    db_obj.close()
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
            db_obj, db_type = get_db()
            if db_type == 'mongo':
                user = db_obj.users.find_one({'email': email, 'pet': pet})
                if user:
                    error = 'Your password: ' + user['password']
                else:
                    error = 'Information not found.'
            else:
                cur = db_obj.cursor()
                cur.execute("SELECT password FROM users WHERE email=? AND pet=?", (email, pet))
                row = cur.fetchone()
                db_obj.close()
                if row:
                    error = 'Your password: ' + row[0]
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
    dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
    
    try:
        db_obj, db_type = get_db()
        if db_type == 'mongo':
            db_obj.predictions.insert_one({
                'date': dt_str,
                'name': user_name,
                'drawing_pred': pred,
                'voice_pred': voicePred,
                'final_pred': final
            })
            records = list(db_obj.predictions.find({'name': user_name}, {'_id': 0}))
            df = pd.DataFrame(records)
        else:
            cur = db_obj.cursor()
            cur.execute("INSERT INTO predictions VALUES (?,?,?,?,?)", (dt_str, user_name, pred, voicePred, final))
            db_obj.commit()
            cur.execute("SELECT * FROM predictions WHERE name=?", (user_name,))
            rows = cur.fetchall()
            db_obj.close()
            df = pd.DataFrame(rows, columns=['Date', 'Name', 'DrawingPrediction', 'VoicePrediction', 'FinalPrediction'])
    except Exception as e:
        print(f"DEBUG: Dashboard error: {e}")
        df = pd.DataFrame(columns=['Date', 'Name', 'DrawingPrediction', 'VoicePrediction', 'FinalPrediction'])

    now_date = now.strftime("%d %B %Y, %H:%M")
    return render_template('dashboard.html',
                           tables=[df.to_html(classes='table-responsive table table-bordered table-hover')],
                           titles=df.columns.values, now_date=now_date)


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
