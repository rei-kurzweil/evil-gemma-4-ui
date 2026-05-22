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

    // Auto-resize textarea
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
        return p; // Return the paragraph element to update it later
    }

    async function handleSend() {
        const text = userInput.value.trim();
        if (!text && !selectedImage) return;

        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // Create the AI message bubble placeholder
        const aiMessageP = appendMessage('ai', '');
        
        try {
            const payload = { message: text };
            if (selectedImage) {
                payload.image = selectedImage.base64;
                payload.image_mime_type = selectedImage.mimeType;
            }

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                aiMessageP.textContent = 'Error: Could not connect to the server.';
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.content) {
                                aiText += data.content;
                                aiMessageP.textContent = aiText;
                                chatWindow.scrollTop = chatWindow.scrollHeight;
                            } else if (data.error) {
                                aiMessageP.textContent = 'Error: ' + data.error;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE line:', e);
                        }
                    }
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
