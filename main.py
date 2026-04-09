from flask import Flask, request, render_template, jsonify
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from flask_bcrypt import Bcrypt
import numpy as np
import pandas as pd
import pickle
import sklearn
import sqlite3
conn = sqlite3.connect("health_history.db")

# Try adding column (will run only once)
try:
    conn.execute("ALTER TABLE history ADD COLUMN username TEXT")
except:
    pass   # already exists → ignore

conn.close()
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
from chatbot import health_chat

# ===================== Version Safety Check =====================
assert sklearn.__version__ >= "1.3.0", "Sklearn version mismatch"

app = Flask(__name__)
app.secret_key = "secret123"
bcrypt = Bcrypt(app)


# ===================== Load CSV Files =====================
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")

# ===================== Load Models =====================
with open("svc.pkl", "rb") as f:
    svc = pickle.load(f)

with open("dt.pkl", "rb") as f:
    dt = pickle.load(f)

# ===================== Load Feature Names =====================
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Use feature names as symptom list (correct source)
all_symptoms = [s.lower() for s in feature_names]

feature_index = {f: i for i, f in enumerate(feature_names)}

print("Model expects features:", len(feature_names))

# ===================== Load BioGPT Model =====================

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
# model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
# def biogpt_chat(question):

#     question = question.strip()

#     prompt = f"Medical question: {question}\nMedical answer:"

#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=120,
#         num_return_sequences=1,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove the prompt from output
#     answer = response.replace(prompt, "").strip()

#     return answer

# ===================== Helper Function =====================
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)

    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ].values.flatten().tolist()

    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrk = workout[workout['disease'] == dis]['workout'].values.tolist()

    return desc, pre, med, die, wrk


# ===================== Prediction Logic =====================
def get_top_predictions(symptoms):

    input_vector = np.zeros(len(feature_names))

    for s in symptoms:
        s = s.strip().lower()
        if s in feature_index:
            input_vector[feature_index[s]] = 1

    # get probabilities
    probs = svc.predict_proba([input_vector])[0]

    # get top 3 indexes
    top3 = probs.argsort()[-3:][::-1]

    results = []

    for i in top3:
        disease = svc.classes_[i]
        probability = round(probs[i] * 100, 2)
        results.append((disease, probability))

    return results

def init_db():

    conn = sqlite3.connect("health_history.db")

    conn.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    symptoms TEXT,
    disease TEXT,
    probability TEXT,
    date TEXT
)
''')

    conn.close()

init_db()

# ===================== Routes =====================


@app.route("/predict", methods=["POST"])
def predict():

    from datetime import datetime

    if 'user' not in session:
        return redirect('/login')

    symptoms = [
        request.form.get("symptom1", ""),
        request.form.get("symptom2", ""),
        request.form.get("symptom3", ""),
        request.form.get("symptom4", "")
    ]

    symptoms = [s.strip().lower() for s in symptoms if s.strip()]

    if not symptoms:
        return render_template("index.html", message="Enter at least one symptom")

    top3 = get_top_predictions(symptoms)

    disease = top3[0][0]
    probability = top3[0][1]

    # Risk level
    if probability >= 70:
        risk_level = "HIGH"
        show_hospitals = True
    elif probability >= 40:
        risk_level = "MEDIUM"
        show_hospitals = False
    else:
        risk_level = "LOW"
        show_hospitals = False

    desc, pre, med, die, wrk = helper(disease)

    symptoms_text = ", ".join(symptoms)

    conn = sqlite3.connect("health_history.db")

    conn.execute(
        "INSERT INTO history(username, symptoms, disease, probability, date) VALUES (?,?,?,?,?)",
        (session['user'], symptoms_text, disease, probability, datetime.now().strftime("%d-%m-%Y %H:%M"))
    )

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        predicted_disease=disease,
        top_predictions=top3,
        risk_level=risk_level,
        show_hospitals=show_hospitals,
        probability=probability,
        dis_des=desc,
        my_precautions=pre,
        medications=med,
        my_diet=die,
        workout=wrk
    )
# -----------------------
# DB CONNECTION
# -----------------------
def get_db():
    return sqlite3.connect("users.db")

# -----------------------
# CREATE TABLE (RUN ONCE)
# -----------------------
def create_table():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT,
                 password TEXT)''')
    conn.close()

create_table()

# -----------------------
# REGISTER
# -----------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        conn = get_db()
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('register.html')

# -----------------------
# LOGIN
# -----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user[2], password):
            session['user'] = username
            return redirect('/dashboard')   # ✅ ADD HERE

        return "Invalid credentials"

    return render_template('login.html')

# -----------------------
# HOME (PROTECTED)
# -----------------------
@app.route("/")
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template("index.html")

# -----------------------
# LOGOUT
# -----------------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    conn = sqlite3.connect("health_history.db")

    data = conn.execute(
        "SELECT symptoms, disease, probability, date FROM history WHERE username=? ORDER BY id DESC",
        (session['user'],)
    ).fetchall()

    conn.close()

    return render_template("history.html", data=data)

# ===================== Suggestion API =====================

@app.route("/suggest")
def suggest():

    query = request.args.get("q", "").lower()

    if not query:
        return jsonify({"suggestions": []})

    suggestions = [s for s in all_symptoms if s.startswith(query)]

    return jsonify({"suggestions": suggestions[:5]})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')

    conn = sqlite3.connect("health_history.db")

    # Total predictions
    total = conn.execute(
        "SELECT COUNT(*) FROM history WHERE username=?",
        (session['user'],)
    ).fetchone()[0]

    # Last prediction
    last = conn.execute(
        "SELECT disease FROM history WHERE username=? ORDER BY id DESC LIMIT 1",
        (session['user'],)
    ).fetchone()

    conn.close()

    return render_template(
        "dashboard.html",
        user=session['user'],
        total=total,
        last=last[0] if last else "No data"
    )
                           
# ===================== Additional Pages =====================

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json.get("message")

    reply = health_chat(user_message)

    return {"reply": reply}

@app.route('/check')
def check():
    conn = sqlite3.connect("health_history.db")
    data = conn.execute("SELECT * FROM history").fetchall()
    conn.close()
    return str(data)
    

# ===================== Run Server =====================

if __name__ == "__main__":
    app.run(debug=True, port=5005)
