import re
import spacy
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found. Entities will be limited.")
            self.nlp = None

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract structured entities from text."""
        entities = {
            "people": [],
            "orgs": [],
            "locations": [],
            "dates": [],
            "money": [],
            "custom": {}
        }
        
        # 1. Regex Extraction (High Precision)
        # Extract Years
        years = re.findall(r"\b20\d{2}\b", text)
        if years:
            entities["custom"]["years"] = list(set(years))

        # Extract Price Thresholds (e.g. $100k, $5.50)
        prices = re.findall(r"\$[\d,]+(?:\.\d+)?(?:k|m|b)?", text, re.IGNORECASE)
        if prices:
             entities["custom"]["prices"] = prices

        # 2. NLP Extraction (General Purpose)
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["people"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["orgs"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
        
        # Deduplicate
        for k in entities:
            if isinstance(entities[k], list):
                entities[k] = list(set(entities[k]))
                
        return entities
