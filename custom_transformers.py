from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Takes in a string for the character to count
# Outputs the number times that character appears in the text
class CountCharacter(BaseEstimator, TransformerMixin):
    def __init__(self, character: str):
        self.character = character

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[text.count(self.character)] for text in X]
    
    
# Define the Advanced Feature Extractor
class SpacyNumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pos_features = []
        ner_features = []
        for doc in self.nlp.pipe(X, batch_size=50):
            counts = {"NOUN": 0, "VERB": 0, "ADJ": 0}
            for token in doc:
                if token.pos_ in counts:
                    counts[token.pos_] += 1
            total = len(doc) + 1e-6
            pos_features.append([counts["NOUN"]/total, counts["VERB"]/total, counts["ADJ"]/total])
            ner_features.append([len(doc.ents)])
        return np.hstack([np.array(pos_features), np.array(ner_features)])
    

# SpacyLemmatizer that removes stopwords and lemmatizes reviews
class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmatized = [
            ' '.join(
                token.lemma_ for token in doc
                if not token.is_stop
            )
            for doc in self.nlp.pipe(X)
        ]
        return lemmatized