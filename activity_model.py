from transformers import TimesformerForVideoClassification, TimesformerFeatureExtractor
from PIL import Image
import torch

class ActivityRecognizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)
        self.feature_extractor = TimesformerFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.labels = self.model.config.label2id
        self.device = device

    def predict(self, frames):
        inputs = self.feature_extractor([Image.fromarray(f) for f in frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
        return list(self.labels.keys())[predicted_label]
