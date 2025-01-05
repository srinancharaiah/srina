import os
import json
import datetime
import csv
import nltk
import ssl
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import streamlit as st

st.set_page_config(page_title="Custom Background", layout="wide")
page_bg_gradient = """ <style> [data-testid="stAppViewContainer"] { background-color: #ADD8E6; } </style> """
st.markdown(page_bg_gradient, unsafe_allow_html=True)

ssl._create_default_https_context = ssl._create_unverified_context

nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

Vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

x = Vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = Vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents['intents']:
        if intent["tag"] == tag:
            response = random.choice(intent['responses'])
            buttons = intent.get('buttons', [])
            return response, buttons

counter = 0

st.markdown(
    """
    <style>
    .chat-button {
        padding: 10px;
        background-color:lightblue;
        color: white;
        border: none;
        cursor: pointer;
        width: 50%;
        text-align: left;
    }
    .chat-button:hover {
        background-color:blue;
    }
    .button-row {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .chat-history-container {
        margin: 0;
        height: 70vh;
        height: 700px;
        overflow-y: scroll;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .chat-row.row-reverse {
        flex-direction: row-reverse;
    }
    .button-container {
    display: flex;
    flex-direction: column; 
    align-items: center; 
    gap: 10px; /* Add spacing between buttons */
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        word-wrap: break-word;
    }
    .human-bubble {
        background-color: #e1f5fe;
        color: #0277bd;
    }
    .ai-bubble {
        background-color: #f1f8e9;
        color: #33691e;
    }
    .fixed-textbox-container {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []
if "buttons" not in st.session_state:
    st.session_state.buttons = {}
if 'clicked_button' not in st.session_state:
    st.session_state.clicked_button = None 
def get_today_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def main():
    global counter
    st.title("Intents of Chatbot using NLP")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        chat_form_key = f"chat-form-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Initialize chat log if not already present
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chat Bot', 'Timestamp'])

        
        if "random_patterns" not in st.session_state:
            st.session_state.random_patterns = random.sample(patterns, min(3, len(patterns)))
        if "counter" not in st.session_state:
            st.session_state.counter = random.randint(1, 10)
        if 'patterns' not in st.session_state:
             st.session_state.patterns = patterns
        if "clicked_radio" not in st.session_state:
            st.session_state.clicked_radio = None
        if "button_counter" not in st.session_state:
            st.session_state.button_counter = 1

        with st.form(chat_form_key, clear_on_submit=True):
            st.markdown('<div class="fixed-textbox-container">', unsafe_allow_html=True)
            cols = st.columns([5, 1])
            user_input = cols[0].text_input("", placeholder="Type your message...", label_visibility="collapsed", key="input_box")
            submit_button = cols[1].form_submit_button("Send", use_container_width=True)
            if user_input:
                submit_button = True
            else:
                submit_button = False
            st.markdown('</div>', unsafe_allow_html=True)

        
        if submit_button:
            timestamp = datetime.datetime.now()
            with open('chat_log.csv', 'a', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
        chat_content = ""
        unique_pairs = set()
        today_date = get_today_date()

        for pattern, response in st.session_state.history:
            user_div = f"""<div class="chat-row row-reverse">
                    <div class="chat-bubble human-bubble">{pattern}</div>
                   </div>"""
            bot_div = f"""<div class="chat-row">
                    <div class="chat-bubble ai-bubble">{response}</div>
                   </div>"""
            chat_content += user_div + bot_div
        if st.session_state.counter:
                num_patterns_to_display = random.randint(2, 4)
                patterns_div = '<div class="chat-row button-container">'
                
                selected_option = st.radio("Select an option:", random.sample(patterns, min(num_patterns_to_display, len(patterns))))
            
                if selected_option:
                        st.session_state.selected_option = selected_option
                if st.session_state.selected_option:
                    response = chatbot(st.session_state.selected_option)[0]
                    st.session_state.history.append((st.session_state.selected_option, response))
                    
                    st.session_state.patterns = random.sample(st.session_state.patterns, len(st.session_state.patterns))

                patterns_div += '</div>'
                chat_content += patterns_div
        for pattern, response in st.session_state.history:
                        user_div = f"""<div class="chat-row row-reverse">
                        <div class="chat-bubble human-bubble">{pattern}</div>
                       </div>"""
            
                        bot_div = f"""<div class="chat-row">
                        <div class="chat-bubble ai-bubble">{response}</div>
                       </div>"""
                        chat_content += user_div + bot_div
                
        st.markdown(f"<div class='chat-history-container'>{chat_content}</div>", unsafe_allow_html=True)

    elif choice == "Conversation History":
        st.write("Today's Conversation History:")
        today_date = get_today_date()
        try:
            with open("chat_log.csv", "r", encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    if len(row) == 3 and row[2].startswith(today_date):
                        st.text(f"User: {row[0]}")
                        st.text(f"Chatbot: {row[1]}")
                        st.text(f"TimeStamp: {row[2]}")
                        st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found for today.")

if __name__ == "__main__":
    main()
