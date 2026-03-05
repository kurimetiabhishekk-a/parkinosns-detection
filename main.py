# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request, session, Response
from werkzeug.utils import secure_filename
from functools import wraps
import sqlite3
import pandas as pd
from datetime import datetime
import os
import time
from utils import *
from voiceTest import *

app = Flask(__name__)
app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True


def init_db():
    """Create all required tables if they don't exist. Safe to call on every startup."""
    con = sqlite3.connect('mydatabase.db')
    cursor = con.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            Date text, Name text, Email text, password text, pet text
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS FinalPred (
            Date text, Name text, DrawingPrediction text,
            VoicePrediction text, FinalPrediction text
        )
    """)
    con.commit()
    con.close()
    # Ensure runtime folders exist
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('upload', exist_ok=True)
    print("DEBUG: Database initialised.")

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
			print(f"DEBUG: Access denied to {request.endpoint}. User not in session. Redirecting to login.")
			return redirect(url_for('login'))
		print(f"DEBUG: Access granted to {request.endpoint} for user {session.get('name')}")
		return f(*args, **kwargs)
	return decorated_function


@app.route('/')
def landing():
	return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
	error = None
	if request.method == 'POST':
		email = request.form.get('email').strip() if request.form.get('email') else ''
		password = request.form.get('password')
		print(f"DEBUG: Login attempt for {email} with password {password}")
		con = sqlite3.connect('mydatabase.db')
		cursor = con.cursor()
		cursor.execute("SELECT Name FROM Users WHERE Email=? AND password=?", (email, password))
		row = cursor.fetchone()
		if row:
			session['name'] = row[0]
			# Initialize prediction session variables
			session['pred'] = 'Healthy'
			session['voicePred'] = 'Healthy'
			print(f"DEBUG: Login successful. Session name set to {session['name']}")
			return redirect(url_for('home'))
		else:
			print("DEBUG: Login failed. Invalid credentials.")
			error = "Invalid Credentials Please try again..!!!"
			return render_template('login.html', error=error)
	return render_template('login.html')


@app.route('/logout')
def logout():
	session.clear()
	return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
	error = None
	if request.method == 'POST' and request.form.get('sub') == 'Submit':
		name = request.form.get('name').strip() if request.form.get('name') else ''
		email = request.form.get('email').strip() if request.form.get('email') else ''
		password = request.form.get('password')
		rpassword = request.form.get('rpassword')
		pet = request.form.get('pet')
		if password != rpassword:
			error = 'Password does not match..!!!'
			return render_template('register.html', error=error)
		con = sqlite3.connect('mydatabase.db')
		cursor = con.cursor()
		cursor.execute("CREATE TABLE IF NOT EXISTS Users (Date text,Name text,Email text,password text,pet text)")
		cursor.execute("SELECT Name FROM Users WHERE Email=?", (email,))
		if cursor.fetchone():
			error = 'User already Registered...!!!'
			return render_template('register.html', error=error)
		now = datetime.now()
		dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
		cursor.execute("INSERT INTO Users VALUES(?,?,?,?,?)", (dt_string, name, email, password, pet))
		con.commit()
		return redirect(url_for('login'))
	return render_template('register.html')


@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
	error = None
	if request.method == 'POST':
		email = request.form.get('email').strip() if request.form.get('email') else ''
		pet = request.form.get('pet').strip() if request.form.get('pet') else ''
		con = sqlite3.connect('mydatabase.db')
		cursor = con.cursor()
		cursor.execute("SELECT password FROM Users WHERE Email=? AND pet=?", (email, pet))
		row = cursor.fetchone()
		if row:
			error = 'Your password : ' + row[0]
		else:
			error = 'Invalid information Please try again..!!!'
		return render_template('forgot-password.html', error=error)
	return render_template('forgot-password.html')


@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
	return render_template('home.html')


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
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	con = sqlite3.connect('mydatabase.db')
	cursor = con.cursor()
	cursor.execute("CREATE TABLE IF NOT EXISTS FinalPred (Date text,Name text,DrawingPrediction text, VoicePrediction text, FinalPrediction text)")
	cursor.execute("INSERT INTO FinalPred VALUES(?,?,?,?,?)", (dt_string, user_name, "Weak Pattern" if pred == 'Parkinson' else pred, "Weak Pattern" if voicePred == 'Parkinson' else voicePred, final))
	con.commit()
	
	conn = sqlite3.connect('mydatabase.db', isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
	df = pd.read_sql_query(f"SELECT * from FinalPred WHERE Name=?", conn, params=(user_name,))
	now_date = now.strftime("%d %B %Y, %H:%M")
	return render_template('dashboard.html', tables=[df.to_html(classes='table-responsive table table-bordered table-hover')], titles=df.columns.values, now_date=now_date)


@app.route('/image', methods=['GET', 'POST'])
@login_required
def image():
	if request.method == 'POST':
		savepath = r'static/img/'
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		# Check for base64 drawing data
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
		
		# Check for file upload
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
		session['pred'] = label  # only update session when a real prediction was made
	return render_template('image_test.html', result=result, suggestion=suggestion, confidence=accuracy)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
	if request.method == 'POST':
		if request.form.get('uploadbutton') == 'Upload':
			savepath = r'upload/'
			f = request.files.get('doc')
			if f:
				f.save(os.path.join(savepath, secure_filename('test.wav')))
				return render_template('upload.html', file=f.filename, mgs='File uploaded..!!')
		elif request.form.get('uploadbutton') == 'Detect PD':
			voice_result = testVoice()
			if len(voice_result) == 2:
				# No file case: returns (label, message)
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


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
