import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
tokenizer = Tokenizer()

try:
    model = load_model('best_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Fit the tokenizer on the text data from the file
try:
    with open('emotion_train.txt', 'r') as file:
        text_data = file.read().splitlines()
        texts = [line.split(';')[0] for line in text_data]
        tokenizer.fit_on_texts(texts)
except Exception as e:
    st.error(f"Error loading or processing text data: {e}")
    st.stop()

# Mapping of predicted class to emotion labels
class_labels = {0: 'Sadness 😭', 1: 'Angry 😠', 2: 'Love 😍', 3: 'Surprise 😲', 4: 'Fear 😨', 5: 'Joy 😂'}

# Inside Streamlit App

# Sidebar for Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page:', ['Home', 'Emotion Classifier'])

if page == 'Home':
    st.title('Welcome to the Emotion Classifier App')

    st.write("""
    This application uses a machine learning model to classify the emotion of a given text.
    The model can predict the following emotions: Sadness, Angry, Love, Surprise, Fear, Joy.
    """)

    st.write('Here is a preview of the dataset used to train the model:')

    # show first 10 rows of dataset
    df = pd.DataFrame(texts, columns=['Text'])
    st.write(df.head(10))

    st.write("""
    ## Model Selection Process

    In the process of selecting the best model for our emotion classifier, we compared three different models: Sequential, RNN, and CNN. 
    Each model was trained and evaluated using the same dataset, and their performance was measured using various metrics such as accuracy, precision, recall, and F1-score.
    The results of these metrics were compiled into a classification report for each model. 
    After thorough evaluation, we found that the CNN model outperformed the other two models in terms of accuracy and overall performance. 
    The classification report for the CNN model showed higher precision and recall values for most of the emotion classes, indicating that it was better at correctly identifying the emotions in the text data.

    ## What is CNN?

    Convolutional Neural Networks (CNNs) are a class of deep learning algorithms that are particularly effective for tasks involving spatial data, such as images and text. CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. This makes them highly effective for tasks like image recognition and text classification.

    In the context of our emotion classifier, the CNN model processes the input text by first converting it into a sequence of word embeddings. These embeddings are then passed through multiple convolutional layers, which extract relevant features from the text. The extracted features are then fed into fully connected layers, which perform the final classification to predict the emotion of the input text.

    The use of CNNs allows our model to capture complex patterns and relationships in the text data, leading to more accurate emotion predictions.
    """)

elif page == 'Emotion Classifier':
    st.title('Text Emotion Classifier')

    # Input text
    input_text = st.text_area('Enter text for classification:')

    if st.button('Classify'):
        if input_text:
            try:
                # Preprocess the input text
                sequences = tokenizer.texts_to_sequences([input_text])
                padded_sequences = pad_sequences(sequences, maxlen=200)  # Assuming max_length is 200

                # Predict the class
                prediction = model.predict(padded_sequences)
                predicted_class = prediction.argmax(axis=1)[0]
                # Get the emotion label
                predicted_emotion = class_labels.get(predicted_class, 'Unknown')

                # Display the result
                st.write(f'Predicted class: {predicted_emotion}')
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write('Please enter some text for classification.')