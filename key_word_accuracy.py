import re
import os
from typing import List

class KeyWordAccuracy():
    def __init__(self, keywords: List[str] = []):
        self.keywords = [
            'change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring', # action
            'music', 'lights', 'volume', 'heat', 'lamp', 'newspaper', 'juice', 'socks', 'shoes', 'Chinese', 'Korean', 'English', 'German', # object
            'kitchen', 'bedroom', 'washroom' # location
        ]
        self.keywords.extend(keywords)
        self.total = 0
        self.correct = 0

    def update(self, predictions: List[str], references: List[str]):
        for pred, ref in zip(predictions, references):
            self.total += 1
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            keyword_in_ref = [word.lower() for word in ref_lower.split() if word in self.keywords]
            keyword_in_pred = [word.lower() for word in pred_lower.split() if word in self.keywords]
            if keyword_in_ref == keyword_in_pred:
                self.correct += 1

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
    
    def clear(self):
        self.total = 0
        self.correct = 0