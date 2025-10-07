from flask import Flask, request, render_template_string, send_file
import requests
import io
import os

app = Flask(__name__)

# Replace with your ElevenLabs API Key
ELEVENLABS_API_KEY = "sk_2595f1f4d65eceda2572614694ca4a7b0c2a44c382dfc90d"

# Sample voice IDs for testing
# (you can find them in your ElevenLabs dashboard or Voices API)
VOICE_IDS = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Bella": "kNie5n4lYl7TrvqBZ4iG"
}

# Home page with a simple form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        voice_id = request.form.get("voice_id")

        # Call ElevenLabs API
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        data = {
            "text": text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return send_file(
                io.BytesIO(response.content),
                mimetype="audio/mpeg",
                as_attachment=False,
                download_name="output.mp3"
            )
        else:
            return f"Error: {response.status_code} - {response.text}"

    # Simple HTML form
    return render_template_string("""
        <h2>🔊 ElevenLabs Voice Test</h2>
        <form method="POST">
            <label>Enter Text:</label><br>
            <textarea name="text" rows="4" cols="50">Hello, this is a test with ElevenLabs!</textarea><br><br>

            <label>Select Voice:</label><br>
            <select name="voice_id">
                {% for name, vid in voices.items() %}
                    <option value="{{ vid }}">{{ name }}</option>
                {% endfor %}
            </select><br><br>

            <button type="submit">Generate Voice</button>
        </form>
    """, voices=VOICE_IDS)


if __name__ == "__main__":
    app.run(debug=True)
