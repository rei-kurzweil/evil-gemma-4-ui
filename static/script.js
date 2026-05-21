document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });

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
        if (!text) return;

        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // Create the AI message bubble placeholder
        const aiMessageP = appendMessage('ai', '');
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
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
        }
    }

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
});
