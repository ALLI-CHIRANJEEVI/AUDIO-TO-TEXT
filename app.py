import os
import whisper
import time
import json
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename
from fpdf import FPDF
from mimetypes import guess_type

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
HISTORY_FILE = "history.json"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Whisper model
model = whisper.load_model("base")

# Load transcription history safely
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
            if isinstance(history, list):
                return history
            else:
                return []
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Save new transcription to history
def save_to_history(filename, text, processing_time, pdf_url):
    history = load_history()
    history_entry = {
        "filename": filename,
        "full_text": text,
        "processing_time": processing_time,
        "pdf_url": pdf_url
    }
    history.insert(0, history_entry)  # Add new item at the beginning
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

# Delete a transcription from history
@app.route("/delete/<filename>", methods=["DELETE"])
def delete_transcription(filename):
    history = load_history()
    updated_history = [entry for entry in history if entry["filename"] != filename]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(updated_history, file, indent=4)

    return jsonify({"message": f"Deleted {filename}"}), 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        # Validate MIME type
        mime_type, _ = guess_type(audio_file.filename)
        if not mime_type or not mime_type.startswith("audio/"):
            return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400

        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(audio_path)

        # Start measuring time
        start_time = time.time()

        # Transcribe audio
        result = model.transcribe(audio_path)
        transcript_text = result["text"]

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)

        # Create PDF
        pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
        pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 10, transcript_text)
        pdf.output(pdf_path)

        # Save to history
        save_to_history(filename, transcript_text, processing_time, f"/download/{pdf_filename}")

        return jsonify({
            "filename": filename,
            "text": transcript_text,
            "pdf_url": f"/download/{pdf_filename}",
            "processing_time": processing_time,
            "history": load_history()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

@app.route("/history")
def get_history():
    return jsonify(load_history())

if __name__ == "__main__":
    app.run(debug=True)
