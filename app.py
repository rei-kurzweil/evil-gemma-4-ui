from flask import Flask, render_template, request, jsonify, Response
from vision_inference import get_model
import json
import os

app = Flask(__name__)

# Initialize model
print("Pre-loading Gemma-4 model...")
try:
    llm_wrapper = get_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    llm_wrapper = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if llm_wrapper is None:
        return jsonify({"response": "Error: Model not loaded on server."}), 500

    data = request.json
    user_message = data.get('message', '')
    image_b64 = data.get('image', None)
    
    def generate():
        try:
            stream = llm_wrapper.generate_response(user_message, image_b64, stream=True)
            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content = delta['content']
                    yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
