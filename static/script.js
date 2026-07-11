document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const modelSelect = document.getElementById('model-select');
    const systemPromptSelect = document.getElementById('system-prompt-select');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const clearImageBtn = document.getElementById('clear-image-btn');
    const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
    let selectedImage = null;
    let availableModels = [];
    let selectedModelId = '';
    let lockedModelId = null;
    let availableSystemPrompts = [];
    let selectedSystemPromptIndex = '';
    let selectedSystemPromptContent = '';

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

    function getSystemPromptByIndex(promptIndex) {
        return availableSystemPrompts.find((prompt) => String(prompt.index) === String(promptIndex)) || null;
    }

    function getModelById(modelId) {
        return availableModels.find((model) => model.id === modelId) || null;
    }

    function selectedModelSupportsVision() {
        const model = getModelById(selectedModelId);
        return Boolean(model && model.capabilities && model.capabilities.supports_vision);
    }

    function updateModelSelectState() {
        modelSelect.disabled = availableModels.length === 0 || Boolean(lockedModelId);
        if (lockedModelId) {
            modelSelect.title = `Model is locked to ${lockedModelId} until the server restarts.`;
            return;
        }
        modelSelect.title = availableModels.length === 0
            ? 'No models available.'
            : 'Select the model used for completions.';
    }

    function renderModelOptions() {
        modelSelect.innerHTML = '';

        if (availableModels.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSelect.appendChild(option);
            updateModelSelectState();
            return;
        }

        for (const model of availableModels) {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.id;
            if (model.loaded) {
                option.textContent += ' (loaded)';
            }
            modelSelect.appendChild(option);
        }

        if (!getModelById(selectedModelId)) {
            selectedModelId = lockedModelId || availableModels[0].id;
        }

        modelSelect.value = selectedModelId;
        updateModelSelectState();
    }

    async function loadModels() {
        try {
            const response = await fetch('/v1/models');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const payload = await response.json();
            availableModels = Array.isArray(payload.data) ? payload.data : [];
            lockedModelId = availableModels.find((model) => model.loaded)?.id || null;
            selectedModelId = lockedModelId || selectedModelId || availableModels[0]?.id || '';
            renderModelOptions();
        } catch (error) {
            console.error('Error loading models:', error);
            availableModels = [];
            lockedModelId = null;
            selectedModelId = '';
            renderModelOptions();
            appendMessage('system', 'Could not load models from the server.');
        }
    }

    function updateSystemPromptSelectState() {
        systemPromptSelect.disabled = availableSystemPrompts.length === 0;
        systemPromptSelect.title = availableSystemPrompts.length === 0
            ? 'No system prompts available.'
            : 'Select the system prompt sent with each completion.';
    }

    function renderSystemPromptOptions() {
        systemPromptSelect.innerHTML = '';

        if (availableSystemPrompts.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No system prompts available';
            systemPromptSelect.appendChild(option);
            updateSystemPromptSelectState();
            return;
        }

        for (const prompt of availableSystemPrompts) {
            const option = document.createElement('option');
            option.value = String(prompt.index);
            option.textContent = prompt.name;
            if (prompt.source === 'workspace') {
                option.textContent += ' (workspace)';
            }
            systemPromptSelect.appendChild(option);
        }

        if (!getSystemPromptByIndex(selectedSystemPromptIndex)) {
            selectedSystemPromptIndex = String(availableSystemPrompts[0].index);
        }

        systemPromptSelect.value = selectedSystemPromptIndex;
        updateSystemPromptSelectState();
    }

    async function loadSystemPromptContent(promptIndex) {
        if (promptIndex === '') {
            selectedSystemPromptContent = '';
            return;
        }

        const response = await fetch(`/system_prompts/${promptIndex}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        selectedSystemPromptContent = typeof payload.content === 'string' ? payload.content : '';
    }

    async function loadSystemPrompts() {
        try {
            const response = await fetch('/system_prompts');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const payload = await response.json();
            availableSystemPrompts = Array.isArray(payload.data) ? payload.data : [];
            if (!getSystemPromptByIndex(selectedSystemPromptIndex) && availableSystemPrompts.length > 0) {
                selectedSystemPromptIndex = String(availableSystemPrompts[0].index);
            }
            renderSystemPromptOptions();
            await loadSystemPromptContent(selectedSystemPromptIndex);
        } catch (error) {
            console.error('Error loading system prompts:', error);
            availableSystemPrompts = [];
            selectedSystemPromptIndex = '';
            selectedSystemPromptContent = '';
            renderSystemPromptOptions();
            appendMessage('system', 'Could not load system prompts from the server.');
        }
    }

    function buildOpenAiMessages(text) {
        const messages = [];

        if (selectedSystemPromptContent) {
            messages.push({ role: 'system', content: selectedSystemPromptContent });
        }

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
            messages.push({ role: 'user', content });
            return messages;
        }

        messages.push({ role: 'user', content: text });
        return messages;
    }

    async function handleSend() {
        const text = userInput.value.trim();
        if (!text && !selectedImage) return;
        if (!selectedModelId) {
            appendMessage('system', 'No model is selected.');
            return;
        }
        if (selectedImage && !selectedModelSupportsVision()) {
            appendMessage('system', `Model '${selectedModelId}' does not support image inputs.`);
            return;
        }

        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        const aiMessageP = appendMessage('ai', '');

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModelId,
                    messages: buildOpenAiMessages(text),
                    stream: true
                })
            });

            if (!response.ok) {
                let errorMessage = 'Could not connect to the server.';
                try {
                    const errorPayload = await response.json();
                    if (errorPayload && errorPayload.error && errorPayload.error.message) {
                        errorMessage = errorPayload.error.message;
                    }
                } catch (error) {
                    console.error('Error parsing completion error:', error);
                }
                aiMessageP.textContent = 'Error: ' + errorMessage;
                await loadModels();
                return;
            }

            if (!response.body) {
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
    modelSelect.addEventListener('change', () => {
        selectedModelId = modelSelect.value;
    });
    systemPromptSelect.addEventListener('change', async () => {
        selectedSystemPromptIndex = systemPromptSelect.value;
        try {
            await loadSystemPromptContent(selectedSystemPromptIndex);
        } catch (error) {
            console.error('Error loading selected system prompt:', error);
            selectedSystemPromptContent = '';
            appendMessage('system', 'Could not load the selected system prompt.');
        }
    });
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    loadModels();
    loadSystemPrompts();
});
