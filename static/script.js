document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const clearImageBtn = document.getElementById('clear-image-btn');
    const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
    let selectedImage = null;

    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });

    function setImagePreview(dataUrl) {
        previewImage.src = dataUrl;
        imagePreview.classList.remove('hidden');
        uploadBtn.classList.add('active');
    }

    function clearSelectedImage() {
        selectedImage = null;
        imageInput.value = '';
        previewImage.removeAttribute('src');
        imagePreview.classList.add('hidden');
        uploadBtn.classList.remove('active');
    }

    function readFileAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Could not read image.'));
            reader.readAsDataURL(file);
        });
    }

    function appendMessage(role, text = '') {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const p = document.createElement('p');
        p.textContent = text;
        msgDiv.appendChild(p);
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return p;
    }

    function buildOpenAiMessages(text) {
        if (selectedImage) {
            const content = [];
            if (text) {
                content.push({ type: 'text', text });
            }
            content.push({
                type: 'image_url',
                image_url: {
                    url: `data:${selectedImage.mimeType};base64,${selectedImage.base64}`
                }
            });
            return [{ role: 'user', content }];
        }

        return [{ role: 'user', content: text }];
    }

    async function handleSend() {
        const text = userInput.value.trim();
        if (!text && !selectedImage) return;

        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        const aiMessageP = appendMessage('ai', '');

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'llama-3.1-local',
                    messages: buildOpenAiMessages(text),
                    stream: true
                })
            });

            if (!response.ok || !response.body) {
                aiMessageP.textContent = 'Error: Could not connect to the server.';
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiText = '';
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

                const events = buffer.split('\n\n');
                buffer = events.pop() || '';

                for (const event of events) {
                    for (const line of event.split('\n')) {
                        if (!line.startsWith('data: ')) {
                            continue;
                        }

                        const dataText = line.slice(6).trim();
                        if (dataText === '[DONE]') {
                            continue;
                        }

                        try {
                            const data = JSON.parse(dataText);
                            const choice = (data.choices || [])[0] || {};
                            const delta = choice.delta || {};

                            if (delta.content) {
                                aiText += delta.content;
                                aiMessageP.textContent = aiText;
                                chatWindow.scrollTop = chatWindow.scrollHeight;
                            } else if (data.error && data.error.message) {
                                aiMessageP.textContent = 'Error: ' + data.error.message;
                            }
                        } catch (error) {
                            console.error('Error parsing SSE line:', error);
                        }
                    }
                }

                if (done) {
                    break;
                }
            }
        } catch (error) {
            console.error('Error:', error);
            aiMessageP.textContent = 'Error: Connection lost.';
        } finally {
            clearSelectedImage();
        }
    }

    uploadBtn.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', async (event) => {
        const [file] = event.target.files;
        if (!file) {
            return;
        }

        if (!file.type.startsWith('image/')) {
            appendMessage('system', 'Only image uploads are supported.');
            clearSelectedImage();
            return;
        }

        if (file.size > MAX_IMAGE_SIZE_BYTES) {
            appendMessage('system', 'Image is too large. Choose one under 10 MB.');
            clearSelectedImage();
            return;
        }

        try {
            const dataUrl = await readFileAsDataUrl(file);
            const [, base64] = dataUrl.split(',', 2);
            selectedImage = {
                name: file.name,
                mimeType: file.type,
                base64
            };
            setImagePreview(dataUrl);
        } catch (error) {
            console.error('Image read error:', error);
            appendMessage('system', 'Could not read the selected image.');
            clearSelectedImage();
        }
    });

    clearImageBtn.addEventListener('click', clearSelectedImage);
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
});
