from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import sklearn
import sqlite3
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
from chatbot import health_chat

# ===================== Version Safety Check =====================
assert sklearn.__version__ >= "1.3.0", "Sklearn version mismatch"

app = Flask(__name__)

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

    conn.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptoms TEXT,
        disease TEXT,
        probability REAL,
        date TEXT
    )
    """)

    conn.close()

init_db()

# ===================== Routes =====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    from datetime import datetime

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

    # risk condition
    show_hospitals = False

    if probability >= 70:
        show_hospitals = True


    # Risk level calculation
    if probability >= 70:
        risk_level = "HIGH"
    elif probability >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    desc, pre, med, die, wrk = helper(disease)

    symptoms_text = ", ".join(symptoms)

    conn = sqlite3.connect("health_history.db")

    conn.execute(
        "INSERT INTO history(symptoms,disease,probability,date) VALUES (?,?,?,?)",
        (symptoms_text, disease, probability, datetime.now().strftime("%d-%m-%Y %H:%M"))
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


@app.route("/history")
def history():

    conn = sqlite3.connect("health_history.db")

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")

    records = cursor.fetchall()

    conn.close()

    return render_template("history.html", records=records)

# ===================== Suggestion API =====================

@app.route("/suggest")
def suggest():

    query = request.args.get("q", "").lower()

    if not query:
        return jsonify({"suggestions": []})

    suggestions = [s for s in all_symptoms if s.startswith(query)]

    return jsonify({"suggestions": suggestions[:5]})


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


# ===================== Run Server =====================

if __name__ == "__main__":
    app.run(debug=True, port=5010)