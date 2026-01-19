import pandas as pd
import numpy as np
import re
import ssl
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK essentials
def download_nltk_data():
    """
    Downloading necessary NLTK datasets
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

download_nltk_data()

class TextPreprocessor:

    """
    Complete Text preprocessing pipeline for Sentiment Analysis
    """

    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemer = PorterStemmer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

    def step1_lowercase(self, text):
        return text.lower()
    
    def step2_remove_html(self,text):
        clean_text = re.sub(r'<.*?>','', text) # this is good <br/> but
        return clean_text

    
    def step3_remove_urls(self,text):
        clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        return clean_text
    
    def step4_remove_punctuation(self,text):
        clean_text = text.translate(str.maketrans('','',string.punctuation))
        return clean_text
    
    def step5_remove_numbers(self,text):
        clean_text = re.sub(r'\d+','', text)
        return clean_text
    
    def step6_remove_extra_spaces(self,text):
        clean_text = ' '.join(text.split())
        return clean_text
    
    def step7_tokenization(self,text):
        tokens = word_tokenize(text)
        return tokens
    
    def step8_remove_stopwords(self,tokens):

        filtered_tokens =  [word for word in tokens if word not in self.stop_words]
        return filtered_tokens
    
    def step9_stemming(self,tokens):
        stemmed_tokens = [self.stemer.stem(word) for word in tokens]
        return stemmed_tokens
    
    def step10_lemmatization(self,tokens):
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens
    
    def preprocess_text(self, text):

        text = self.step1_lowercase(text)

        text = self.step2_remove_html(text)

        text = self.step3_remove_urls(text)

        text = self.step4_remove_punctuation(text)

        text = self.step5_remove_numbers(text)

        text = self.step6_remove_extra_spaces(text)

        tokens = self.step7_tokenization(text) 

        tokens = self.step8_remove_stopwords(tokens)

        if self.use_stemming:
            tokens = self.step9_stemming(tokens)

        if self.use_lemmatization:
            tokens = self.step10_lemmatization(tokens)
        
        clean_text = ' '.join(tokens)

        return clean_text
    
    def preprocess_dataframe(self, df, text_column='review'):
        
        df['cleaned_text'] = df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )

        return df
    
if __name__ == "__main__":

    sample_text = """
    After watching this movie I was honestly disappointed - not because of the actors, story or directing - I was disappointed by this film advertisements.<br /><br />The trailers were suggesting that the battalion "have chosen the third way out" other than surrender or die (Polish infos were even misguiding that they had the choice between being killed by own artillery or German guns, they even translated the title wrong as "misplaced battalion"). This have tickled the right spot and I bought the movie.<br /><br />The disappointment started when I realized that the third way is to just sit down and count dead bodies followed by sitting down and counting dead bodies... Then I began to think "hey, this story can't be that simple... I bet this clever officer will find some cunning way to save what left of his troops". Well, he didn't, they were just sitting and waiting for something to happen. And so was I.<br /><br />The story was based on real events of World War I, so the writers couldn't make much use of their imagination, but even thought I found this movie really unchallenging and even a little bit boring. And as I wrote in the first place - it isn't fault of actors, writers or director - their marketing people have raised my expectations high above the level that this movie could cope with.
"""

    preprocessor = TextPreprocessor(use_lemmatization=True)

    cleaned = preprocessor.preprocess_text(sample_text)

    print(cleaned)



    

