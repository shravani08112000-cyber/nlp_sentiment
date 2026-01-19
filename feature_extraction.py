import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureExtractor:

    """
    Feature extraction methods using BoW and TF-IDF
    """

    def __init__(self, method='tfidf',max_features=5000):

        self.method = method
        self.max_features = max_features
        self.vectorizer = None

   
    def fit_transform(self, documents):
       
        """Fit and transform document to feature vectors"""

        if self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
       
        feature_matrix = self.vectorizer.fit_transform(documents)

        print(feature_matrix.shape)
        print(len(self.vectorizer.vocabulary_))

        return feature_matrix
   
    def transform(self, documents):
        """Transform new documents using the appropriate vectoriser"""

        if self.vectorizer is None:
            raise ValueError("vectoriser not fitted yet, call the fit_transofrm intitally")
       
        return self.vectorizer.transform(documents)
   
    def get_feature_names(self):

        """Get feature names"""

        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()
   

if __name__ == "__main__":
    sample_docs = [
        "this movie is super boring",
        "Amazing movie to watch and I recommend this to everyone",
        "Terrible to watch this movie",
        "The protogonist played an amzing role and wonderful to watch this movie"
    ]

    bow = FeatureExtractor(method='bow',max_features=50)
    bow_matrix = bow.fit_transform(sample_docs)
    print(bow.get_feature_names()[:10])

    tfidf = FeatureExtractor(method='tfidf',max_features=50)
    tfidf_matric = tfidf.fit_transform(sample_docs)
    print(tfidf.get_feature_names()[:10])
