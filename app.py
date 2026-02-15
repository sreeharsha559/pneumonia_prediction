import os
import json
import sqlite3
import hashlib
from datetime import datetime

import numpy as np
import cv2
import qrcode

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import request

import firebase_admin
from firebase_admin import credentials, auth

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- Firebase Init ----------------
def initialize_firebase():
    if not firebase_admin._apps:
        firebase_key = os.environ.get("FIREBASE_KEY")

        if firebase_key:
            firebase_config = json.loads(firebase_key)
            cred = credentials.Certificate(firebase_config)
        else:
            # Local development fallback
            cred = credentials.Certificate("firebase_key.json")

        firebase_admin.initialize_app(cred)

# Call it once
initialize_firebase()

# ---------------- Database Init ----------------
def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            mobile TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT,
            report_file TEXT,
            hash TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- Load AI Model ----------------
model = load_model("model.h5")
IMG_SIZE = (224, 224)

# ---------------- GradCAM ----------------
def generate_gradcam(model, img_array, last_conv_layer_name):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap(heatmap, original_img_path, output_path):
    img = cv2.imread(original_img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(output_path, superimposed_img)

# ---------------- PDF Generation ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

def generate_pdf_report(patient_name, age, mobile,
                        result, confidence,
                        original_path, heatmap_path,
                        record_hash):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    file_name = f"{mobile}.pdf"
    file_path = os.path.join("static", file_name)

    # Generate QR
    base_url = request.host_url
    qr_data = f"{base_url}verify/{record_hash}"

    qr = qrcode.make(qr_data)
    qr_path = os.path.join("static", f"qr_{mobile}.png")
    qr.save(qr_path)


    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Heading1"]

    elements.append(Paragraph("AI-Assisted Pneumonia Detection Report", title))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Generated On: {timestamp}", normal))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Patient Name: {patient_name}", normal))
    elements.append(Paragraph(f"Age: {age}", normal))
    elements.append(Paragraph(f"Mobile: {mobile}", normal))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Prediction: {result}", normal))
    elements.append(Paragraph(f"Confidence: {confidence}%", normal))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Original X-ray:", normal))
    elements.append(Spacer(1, 10))
    elements.append(Image(original_path, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Grad-CAM Heatmap:", normal))
    elements.append(Spacer(1, 10))
    elements.append(Image(heatmap_path, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Verification Hash:", normal))
    elements.append(Paragraph(record_hash, normal))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Scan QR for digital verification:", normal))
    elements.append(Spacer(1, 10))
    elements.append(Image(qr_path, width=2*inch, height=2*inch))

    doc.build(elements)

    return file_name

# ---------------- Routes ----------------

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    id_token = request.json.get("idToken")
    try:
        decoded_token = auth.verify_id_token(id_token)
        session["user"] = decoded_token["uid"]
        return jsonify({"status": "success"})
    except:
        return jsonify({"status": "error"}), 401


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login_page"))

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("index.html")

@app.route("/verify/<record_hash>")
def verify_record(record_hash):
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name, age, mobile, result, confidence, timestamp FROM patients WHERE hash = ?", (record_hash,))
    data = cursor.fetchone()

    conn.close()

    if data:
        return render_template(
            "verify.html",
            name=data[0],
            age=data[1],
            mobile=data[2],
            result=data[3],
            confidence=data[4],
            timestamp=data[5]
        )
    else:
        return "Record not found"


@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login_page"))

    patient_name = request.form["patient_name"]
    age = request.form["age"]
    mobile = request.form["mobile"]

    file = request.files["file"]
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    confidence_score = float(prediction)

# Reject if model is unsure
    if 0.4 < confidence_score < 0.6:
        return render_template(
            "index.html",
            error="Not a valid chest X-ray image. Please upload a proper lung X-ray."
        )
    
    confidence = round(confidence_score * 100, 2)
    
    if confidence_score > 0.5:
        result = "Pneumonia Detected"
    else:
        result = "Normal"
        confidence = round(100 - confidence, 2)

    # Generate hash
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record_string = f"{patient_name}{age}{mobile}{result}{confidence}{timestamp}"
    record_hash = hashlib.sha256(record_string.encode()).hexdigest()

    # GradCAM
    heatmap = generate_gradcam(model, img_array, "conv5_block3_out")
    heatmap_filename = "heatmap_" + file.filename
    heatmap_path = os.path.join("static", heatmap_filename)
    overlay_heatmap(heatmap, filepath, heatmap_path)

    # Generate PDF
    report_file = generate_pdf_report(
        patient_name, age, mobile,
        result, confidence,
        filepath, heatmap_path,
        record_hash
    )

    # Store in DB
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patients (name, age, mobile, result, confidence, timestamp, report_file, hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, age, mobile, result, confidence, timestamp, report_file, record_hash))

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        original_image=file.filename,
        heatmap_image=heatmap_filename,
        report_file=report_file
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
