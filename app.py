import os
import json
import csv
import streamlit as st
import nltk
import random
import datetime
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

class AdvancedChatbot:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = os.path.abspath(intents_file)
        with open(self.intents_file, "r") as file:
            self.intents_data = json.load(file)

        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 4))
        self.prepare_training_data()
        self.train_classifier()
        self.init_conversation_log()
        self.counter = 0

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def prepare_training_data(self):
        self.patterns = []
        self.tags = []

        for intent in self.intents_data:
            for pattern in intent['patterns']:
                preprocessed_pattern = self.preprocess_text(pattern)
                self.patterns.append(preprocessed_pattern)
                self.tags.append(intent['tag'])

    def train_classifier(self):
        X = self.vectorizer.fit_transform(self.patterns)
        y = self.tags

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

    def chatbot_response(self, input_text):
        input_vectorized = self.vectorizer.transform([input_text])

        tag = self.classifier.predict(input_vectorized)[0]

        for intent in self.intents_data:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                self.log_conversation(input_text, response)
                return response

        return "I'm not sure how to respond to that."

    def init_conversation_log(self):
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    def log_conversation(self, user_input, response):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

def main():
    chatbot = AdvancedChatbot()

    st.title("Advanced NLP Chatbot")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        chatbot.counter += 1

        user_input = st.text_input("You:", key=f"user_input_{chatbot.counter}")

        if user_input:
            response = chatbot.chatbot_response(user_input)

            st.text_area("Chatbot:", value=response, height=120, key=f"chatbot_response_{chatbot.counter}")

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")

        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("An advanced chatbot using Natural Language Processing (NLP) and Machine Learning.")

        st.subheader("Project Overview:")
        st.write("""
        The chatbot uses:
        1. NLP techniques for intent classification
        2. Machine Learning for response generation
        3. Streamlit for interactive web interface
        """)

        st.subheader("Technical Details:")
        st.write("""
        - Random Forest Classifier for intent detection
        - TF-IDF Vectorization
        - NLTK for text preprocessing
        - Conversation logging
        """)

if __name__ == "__main__":
    main()
