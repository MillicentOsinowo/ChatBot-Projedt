import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

lemmatizer = nltk.stem.WordNetLemmatizer()
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

df = pd.DataFrame()
df['Questions'] = cust[1].reset_index(drop = True)
df['Answer'] = sales[1].reset_index(drop = True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        # The code above does the following:
        # Identifies every word in the sentence
        # Turns it to a lower case
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    return ' '.join(preprocessed_sentences)
df['tokenized Questions'] = df['Questions'].apply(preprocess_text)


corpus = df['tokenized Questions'].to_list()
tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus)

def get_response(user_input):
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vector.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, v_corpus) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return df['Answer'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response


# create greeting list
import random
chatbot_greetings = ["Hello there, Welcome to Samsung Bot.... How can I help",
            "Hi Human.... How can I help",
            'How far Dear User, wetin happen'
            "Good Day .... How can I help",
            "Hello There... How can I be useful to you today",
            "Hi Student.... How can I be of use"]

user_greeting = ["Hi","Hello","Hey","Hello ther"]
exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']

# Test your chatbot
#while True:
#    user_input = st.text_input("You: ")
 #   if user_input.lower() in exits:
  #      print(f"\nChatbot: {random_farewell}!")
   #     break
    #if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
     #   print(f"\nChatbot: {random_greetings}!")
   # else:
    #    response = get_response(user_input)
     #   st.write(f"\nChatbot: {response}")


# --------------------- STREAMLIT IMPLEMENTATION  ------------
st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG CUSTOMER CARE</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY MTW Gomycode</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('pngwing.com.png', caption = 'Samsung Products Related Chats')


def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return df['Answer'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to Orpheus Bot. pls ejoy your usage",
    "Hi user, This bot is created by oprheus, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


user_q = col2.text_input('Pls make your enquiries about Samsung Products: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = bot_response(user_q)
    col2.write(f'ChatBot:  {responses}')