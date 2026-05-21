from flask import Flask, render_template, request, jsonify
from vision_inference import get_model
import os

app = Flask(__name__)

# Initialize model (this might take a few seconds)
# In production, you might want to lazy-load this or use a separate worker
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
    image_b64 = data.get('image', None) # Placeholder for when we add vision
    
    try:
        response_text = llm_wrapper.generate_response(user_message, image_b64)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"response": f"Sorry, an error occurred during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the network
    app.run(debug=False, host='0.0.0.0', port=5000)
