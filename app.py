from flask import Flask, render_template, request, jsonify, Response
from analyzer import consume_complete_sentences
from demultiplexer import Demultiplexer
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
        response_parts = []
        pending_text = ""
        demultiplexer = Demultiplexer()
        try:
            stream = llm_wrapper.generate_response(user_message, image_b64, stream=True)
            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content = delta['content']
                    response_parts.append(content)
                    pending_text += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

                    complete_sentences, pending_text = consume_complete_sentences(pending_text)
                    if complete_sentences:
                        for sentence in complete_sentences:
                            demultiplexer.route_sentence(sentence)

            if pending_text.strip():
                demultiplexer.route_sentence(pending_text.strip())

            print("\n\n\n")
            print(demultiplexer.pretty_print())
            print("\n\n\n")
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
