const API_BASE = '';
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');

let sessionId = null;

async function createSession() {
    try {
        const response = await fetch(`${API_BASE}/session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        sessionId = data.session_id;
    } catch (error) {
        console.error('Failed to create session:', error);
    }
}

async function sendQuestion(question) {
    try {
        const response = await fetch(`${API_BASE}/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response');
        }
        
        const data = await response.json();
        
        if (data.session_id) {
            sessionId = data.session_id;
        }
        
        return data;
    } catch (error) {
        console.error('Error sending question:', error);
        throw error;
    }
}

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = content;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant loading';
    loadingDiv.id = 'loading-message';
    loadingDiv.textContent = 'Typing...';
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();
}

function removeLoadingMessage() {
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

function scrollToBottom() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function setInputEnabled(enabled) {
    questionInput.disabled = !enabled;
    sendButton.disabled = !enabled;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question) return;
    
    questionInput.value = '';
    
    addMessage(question, 'user');
    setInputEnabled(false);
    addLoadingMessage();
    
    try {
        const result = await sendQuestion(question);
        
        removeLoadingMessage();
        
        if (result.answer) {
            addMessage(result.answer, 'assistant');
        } else {
            addMessage('Sorry, ik kon geen antwoord genereren. Probeer het opnieuw.', 'error');
        }
    } catch (error) {
        removeLoadingMessage();
        addMessage('Er is een fout opgetreden. Controleer of de server draait en probeer opnieuw.', 'error');
    }
    
    setInputEnabled(true);
    questionInput.focus();
});

async function init() {
    await createSession();
    
    questionInput.focus();
}

init();