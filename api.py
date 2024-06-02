from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline
import os
import json
from pydub import AudioSegment
import logging

app = Flask(__name__)
CORS(app)

@app.route('/api/ml', methods=['POST'])
def run_asr():
    if 'videoFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    video_file = request.files['videoFile']
    video_file_path = 'temp_video_file.mp4'
    video_file.save(video_file_path)

    audio_file_path = 'temp_audio_file.wav'

    try:
        # Extract audio using pydub
        audio = AudioSegment.from_file(video_file_path)
        audio.export(audio_file_path, format="wav")

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
        )

        result = asr_pipeline(audio_file_path, generate_kwargs={"max_new_tokens": 256}, return_timestamps=True)

        logging.debug(f"ASR result: {result}")

        text = result['text']
        chunks = result['chunks']

        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

    return jsonify({"text": text, "chunks": chunks})

if __name__ == "__main__":
    app.run(debug=True)
