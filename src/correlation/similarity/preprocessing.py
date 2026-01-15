import re
import spacy
from typing import List

class TextPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            self.nlp = None

        self.stop_words = {
            "the", "a", "an", "is", "are", "be", "will", "on", "in", "at", "by", "for", "to", "of"
        }
        
        # Domain specific normalization
        self.replacements = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "donald trump": "trump",
            "joe biden": "biden",
            "kamala harris": "harris"
        }

    def normalize(self, text: str) -> str:
        """
        Normalize text: lowercase, remove punctuation, expand abbreviations.
        """
        if not text:
            return ""
            
        text = text.lower()
        
        # Replace common terms
        for k, v in self.replacements.items():
            text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
            
        # Remove punctuation/special chars but keep numbers and $
        text = re.sub(r'[^a-z0-9\s$]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """Split into meaningful tokens."""
        text = self.normalize(text)
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            return tokens
        else:
            # Fallback if spacy fails
            return [w for w in text.split() if w not in self.stop_words]
