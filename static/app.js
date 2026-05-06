const API_BASE = '';
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');

let sessionId = null;
let currentLang = 'nl'; // Default: Dutch

const translations = {
    nl: {
        title: 'Medicatie Q&A Assistent',
        subtitle: 'Stel vragen over uw medicatie',
        welcome: 'Hallo! Ik ben uw virtuele assistent voor vragen over medicatie.',
        example: 'Stel gerust uw vraag.',
        placeholder: 'Typ uw vraag hier...',
        loading: 'Typing...',
        error: 'Sorry, ik kon geen antwoord genereren. Probeer het opnieuw.',
        errorServer: 'Er is een fout opgetreden. Controleer of de server draait en probeer opnieuw.',
        disclaimer: 'Dit is een AI-assistent en geen vervanging voor professioneel medisch advies. Raadpleeg altijd uw arts of apotheker.'
    },
    en: {
        title: 'Medication Q&A Assistant',
        subtitle: 'Ask questions about your medication',
        welcome: 'Hello! I am your virtual assistant for medication questions.',
        example: 'Feel free to ask a question.',
        placeholder: 'Type your question here...',
        loading: 'Typing...',
        error: 'Sorry, I could not generate an answer. Please try again.',
        errorServer: 'An error occurred. Check if the server is running and try again.',
        disclaimer: 'This is an AI assistant and not a replacement for professional medical advice. Always consult your doctor or pharmacist.'
    }
};

function getButtonLabel() {
    // Button shows the language it will switch TO
    return currentLang === 'nl' ? 'English' : 'Nederlands';
}

function updateUILanguage() {
    // Update HTML lang attribute
    document.documentElement.lang = currentLang;
    
    // Update page title
    document.title = t('title');
    
    // Update header title and subtitle
    const titleEl = document.querySelector('.app-header h1');
    const subtitleEl = document.querySelector('.app-header .subtitle');
    if (titleEl) titleEl.textContent = t('title');
    if (subtitleEl) subtitleEl.textContent = t('subtitle');
    
    // Update welcome message
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.innerHTML = `
            <p>${t('welcome')}</p>
            <p>${t('example')}</p>
        `;
    }
    
    // Update placeholder
    questionInput.placeholder = t('placeholder');
    
    // Update disclaimer
    const disclaimerP = document.querySelector('.disclaimer p');
    if (disclaimerP) disclaimerP.textContent = t('disclaimer');
    
    // Update button to show the OTHER language (what it will switch TO)
    document.getElementById('lang-switch').textContent = getButtonLabel();
}

function t(key) {
    return translations[currentLang][key] || key;
}

function switchLanguage() {
    currentLang = currentLang === 'nl' ? 'en' : 'nl';
    updateUILanguage();
    // Store preference
    localStorage.setItem('preferredLang', currentLang);
    console.log('Language switched to:', currentLang);
}

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
        console.log('Sending question with language:', currentLang);
        const response = await fetch(`${API_BASE}/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: sessionId,
                language: currentLang
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
            addMessage(t('error'), 'error');
        }
    } catch (error) {
        removeLoadingMessage();
        addMessage(t('errorServer'), 'error');
    }
    
    setInputEnabled(true);
    questionInput.focus();
});

async function init() {
    // Load saved language preference
    const savedLang = localStorage.getItem('preferredLang');
    if (savedLang && (savedLang === 'nl' || savedLang === 'en')) {
        currentLang = savedLang;
    }
    updateUILanguage();
    
    await createSession();
    
    questionInput.focus();
    
    console.log('Initialized with language:', currentLang);
}

init();