# Implementation of Chatbot using NLP

## Overview
An intelligent chatbot built with Python, leveraging advanced Natural Language Processing (NLP) and Machine Learning techniques for sophisticated conversational interactions.

## Features
- Advanced intent classification
- Text preprocessing
- Machine learning-powered response generation
- Conversation logging
- Streamlit web interface
- Multiple interaction modes (Home, Conversation History, About)

## Prerequisites
- Python 3.8+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dqrshan/Implementation-of-Chatbot-using-NLP.git
cd Implementation-of-Chatbot-using-NLP
```

2. Create a virtual environment: (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

## Usage
Run the chatbot:
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Main chatbot application
- `intents.json`: Intent configuration file (Provided by TechSaksham)
- `requirements.txt`: Project dependencies
- `chat_log.csv`: Conversation history log

## Technologies
- Streamlit
- scikit-learn
- NLTK
- Pandas
- Random Forest Classifier
- TF-IDF Vectorization

## License
This project is licensed under the [MIT License](https://github.com/Dqrshan/Implementation-of-Chatbot-using-NLP/blob/main/LICENSE).
