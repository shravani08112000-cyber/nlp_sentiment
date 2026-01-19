

from data_ingestion import load_imdb_data
from preprocessing import TextPreprocessor
from model_training import SentimentModel
from feature_extraction import FeatureExtractor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle


class SentimentAnalysisPipeline:


    def __init__(self,
                use_lemmatization=True,
                feature_method='tfidf',
                max_features=5000,
                model_type="logistic_regression"):
        
        self.preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
        self.feature_extractor = FeatureExtractor(
            method=feature_method,
            max_features=max_features
        )
        self.model = SentimentModel(model_type=model_type)


    def run_complete_pipeline(self,sample_size=5000,test_size=0.2):

        #step 1
        train_df, test_df = load_imdb_data(sample_size=sample_size)

        # combine data
        df = pd.concat([train_df,test_df],ignore_index=True)


        #preprocessing
        df = self.preprocessor.preprocess_dataframe(df)

        X = self.feature_extractor.fit_transform(
            df['cleaned_text'].tolist()
        )

        y = df['label'].values

        #slpit the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Model training

        self.model.train(X_train, y_train)

        metrics = self.model.evaluate(X_test, y_test)
        
        self.is_trained = True

        self.save_pipeline('sentiment_model.pkl')

        return metrics

    def predict_sentiment(self,text):

        if not self.is_trained:
            raise ValueError("Model Not Trained Yet So Run the pipeline")
        
        #preprocess
        cleaned_text = self.preprocessor.preprocess_text(text)

        # features
        features = self.feature_extractor.transform([cleaned_text])

        # predict

        prediction = self.model.predict(features)[0]

        try:
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        except:
            confidence = None

        return prediction, confidence

    def save_pipeline(self, filepath):

        pipeline_data = {
            'preprocessor':self.preprocessor,
            'feature_extractor':self.feature_extractor,
            'model': self.model,
            'is_trained': self.is_trained
        }

        with open(filepath, "wb") as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipline saved to {filepath}")
    
    def load_pipeline(self, filepath):

        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.preprocessor = pipeline_data['preprocessor']
        self.feature_extractor = pipeline_data['feature_extractor']
        self.model = pipeline_data['model']
        self.is_trained = pipeline_data['is_trained']

        print(f"Pipline loaded from {filepath}")
    
    def predict_batch(self, texts):

        if not self.is_trained:
            raise ValueError("Model Not Trained Yet So Run the pipeline")
        
        cleaned_text = [self.preprocessor.preprocess_text(text) for text in texts]

        features = self.feature_extractor.transform(cleaned_text)

        predictions = self.model.predict(features)

        try:
            probabilities = self.model.predict_proba(features)
        except:
            probabilities = None

        return predictions, probabilities



if __name__ == "__main__":

    pipeline = SentimentAnalysisPipeline(
            use_lemmatization=True,
            feature_method='tfidf',
            max_features=5000,
            model_type="logistic_regression")
    
    #metrics = pipeline.run_complete_pipeline(sample_size=1000)

     # Run pipeline with sample data for faster execution
    metrics = pipeline.run_complete_pipeline(sample_size=2000)






