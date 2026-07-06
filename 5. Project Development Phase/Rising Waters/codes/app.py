"""
Rising Waters - Flask Web Application with Authentication & History
======================================================================
Serves pages:
  /               -> home.html (Landing Page - Protected)
  /predict        -> index.html (Input Form Page - Protected)
  /Predict        -> index.html (Input Form Page - Protected)
  /chance         -> chance.html (Flood predicted result page - Protected)
  /no_chance      -> no_chance.html (No flood predicted result page - Protected)
  /analysis       -> analysis.html (EDA Charts & Analysis - Protected)
  /history        -> history.html (Prediction History Log - Protected)
  /login          -> login.html (Login interface)
  /register       -> register.html (Registration/Signup)
  /logout         -> Logs the user out
"""

from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.url_map.strict_slashes = False
app.secret_key = 'rising_waters_secret_key'  # Necessary for session storage and security

DATABASE = "database.db"

# Lazy-load model & scaler
_model = None
_scaler = None

def get_model():
    global _model, _scaler
    if _model is None or _scaler is None:
        _model = None
        _scaler = None
        # Try root paths first (matching model = load('floods.save'))
        if os.path.exists("floods.save") and os.path.exists("transform.save"):
            try:
                _model = joblib.load("floods.save")
                _scaler = joblib.load("transform.save")
            except Exception:
                _model, _scaler = None, None
        
        if _model is None or _scaler is None:
            if os.path.exists("floods.save") and os.path.exists("scaler.save"):
                try:
                    _model = joblib.load("floods.save")
                    _scaler = joblib.load("scaler.save")
                except Exception:
                    _model, _scaler = None, None

        if _model is None or _scaler is None:
            # Fallback to model directory
            base = os.path.dirname(os.path.abspath(__file__))
            m_path1 = os.path.join(base, "floods.save")
            s_path1 = os.path.join(base, "transform.save")
            m_path2 = os.path.join(base, "model", "floods.save")
            s_path2 = os.path.join(base, "model", "scaler.save")

            if os.path.exists(m_path1) and os.path.exists(s_path1):
                try:
                    _model = joblib.load(m_path1)
                    _scaler = joblib.load(s_path1)
                except Exception:
                    _model, _scaler = None, None
            elif os.path.exists(m_path2) and os.path.exists(s_path2):
                try:
                    _model = joblib.load(m_path2)
                    _scaler = joblib.load(s_path2)
                except Exception:
                    _model, _scaler = None, None
    return _model, _scaler

# Database Access Helpers
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # Create users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        # Create history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                temp REAL,
                humidity REAL,
                cloud_cover REAL,
                annual REAL,
                jan_feb REAL,
                mar_may REAL,
                jun_sep REAL,
                oct_dec REAL,
                prediction INTEGER,
                probability REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()

# Authentication Guard Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# Setup Database on Import/Startup
init_db()

FEATURES = ["Temp", "Humidity", "Cloud Cover", "ANNUAL",
            "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]

# ─── Authentication Routes ───

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register/Signup option before logging in."""
    if "user_id" in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            return render_template("register.html", error="All fields are required.")

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")

        try:
            pw_hash = generate_password_hash(password)
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, pw_hash)
                )
                conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username or Email already registered.")
            
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login with username/email and password credentials."""
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password")

        if not login_id or not password:
            return render_template("login.html", error="All fields are required.")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (login_id, login_id)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["email"] = user["email"]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username/email or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clear user session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))

# ─── Protected Project Routes ───

@app.route("/")
@login_required
def home():
    """Landing page explaining the project."""
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
@app.route("/Predict", methods=["GET"])
@login_required
def predict_form():
    """Input form for weather parameters."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/Predict", methods=["POST"])
@login_required
def predict_result():
    """Predict logic - handles form inputs, scales, logs to DB, and redirects."""
    model, scaler = get_model()
    if model is None:
        return render_template("index.html",
                               error="Model files not found. Please run train_model.py first.")
    try:
        # Flexible parser supporting both 5-feature form and 8-feature form
        temp = float(request.form.get("Temp") or request.form.get("temperature") or 29.5)
        humidity = float(request.form.get("Humidity") or request.form.get("humidity") or 78.2)
        cloud_cover = float(request.form.get("Cloud Cover") or request.form.get("cloud_cover") or 0.0)
        annual = float(request.form.get("ANNUAL") or request.form.get("annual") or request.form.get("Annual Rain Fall") or 0.0)
        jan_feb = float(request.form.get("Jan-Feb") or request.form.get("jan_feb") or request.form.get("Jan-Feb Rainfall") or 0.0)
        mar_may = float(request.form.get("Mar-May") or request.form.get("mar_may") or request.form.get("March-May Rainfall") or 0.0)
        jun_sep = float(request.form.get("Jun-Sep") or request.form.get("jun_sep") or request.form.get("June-September") or 0.0)
        oct_dec = float(request.form.get("Oct-Dec") or request.form.get("oct_dec") or 180.2)

        arr = np.array([temp, humidity, cloud_cover, annual, jan_feb, mar_may, jun_sep, oct_dec]).reshape(1, -1)
        arr_sc = scaler.transform(arr)
        pred = model.predict(arr_sc)[0]
        prob = model.predict_proba(arr_sc)[0]
        flood_prob = round(float(prob[1]) * 100, 2)

        # Store in user session
        session['probability'] = flood_prob
        session['form_data'] = {
            "Temperature": temp,
            "Humidity": humidity,
            "Cloud Cover": cloud_cover,
            "Annual Rainfall": annual,
            "Jan-Feb": jan_feb,
            "Mar-May": mar_may,
            "Jun-Sep": jun_sep,
            "Oct-Dec": oct_dec
        }

        # Log prediction to database history
        with get_db() as conn:
            conn.execute("""
                INSERT INTO history (
                    user_id, temp, humidity, cloud_cover, annual,
                    jan_feb, mar_may, jun_sep, oct_dec, prediction, probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"], temp, humidity, cloud_cover, annual,
                jan_feb, mar_may, jun_sep, oct_dec, int(pred), flood_prob
            ))
            conn.commit()

        if pred == 1:
            return redirect(url_for('chance'))
        else:
            return redirect(url_for('no_chance'))
    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/chance")
@login_required
def chance():
    """Alert page displayed if a flood is predicted."""
    probability = session.get('probability', 0.0)
    form_data = session.get('form_data', {})
    return render_template("chance.html", probability=probability, form_data=form_data)


@app.route("/no_chance")
@login_required
def no_chance():
    """Safe page displayed if no flood is predicted."""
    probability = session.get('probability', 0.0)
    form_data = session.get('form_data', {})
    return render_template("no_chance.html", probability=probability, form_data=form_data)


@app.route("/analysis")
@login_required
def analysis():
    """EDA & Model Performance charts page."""
    base = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base, "static", "images")
    charts = {
        "univariate_dist": "Univariate Analysis — Feature Distributions",
        "boxplots":        "Box Plots — Outlier Detection by Class",
        "heatmap":         "Correlation Heatmap — Multivariate Analysis",
        "pairplot":        "Pair Plot — Multivariate Relationships",
        "class_dist":      "Target Class Distribution",
        "model_comparison":"Model Accuracy Comparison",
        "confusion_matrix":"XGBoost Confusion Matrix",
    }
    available = {k: v for k, v in charts.items()
                 if os.path.exists(os.path.join(img_dir, f"{k}.png"))}
    trained = os.path.exists("floods.save") or os.path.exists(os.path.join(base, "model", "floods.save"))
    return render_template("analysis.html",
                           charts=available,
                           trained=trained)


@app.route("/history")
@login_required
def history():
    """History of who used this analysis, showing user identity and details."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT h.*, u.username, u.email
            FROM history h
            JOIN users u ON h.user_id = u.id
            ORDER BY h.timestamp DESC
        """).fetchall()
    return render_template("history.html", history=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
