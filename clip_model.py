
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor, CLIPProcessor
import numpy as np

"""
Creates instance of CLIP model, image processor and text tokenizer as specified 
by model_id constructor parameter.
Includes methods to encode images and texts.
"""
class CLIP:
    def __init__(self, model_id):
        self.image_processor = CLIPImageProcessor.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.combined_processor = CLIPProcessor.from_pretrained(model_id)
    

    def encode_image(self, images):
        inputs = self.image_processor(images, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs).detach().numpy()

        # Normalize  
        norm = []
        for x in image_features:
            norm.append(x / np.linalg.norm(x))
        return np.array(norm)

    
    def encode_text(self, texts):
         inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
         text_features = self.model.get_text_features(**inputs).detach().numpy()

         # Normalize  
         norm = []
         for x in text_features:
            norm.append(x / np.linalg.norm(x))
         return np.array(norm)

    def encode_both(self, imgs, txts):
        inputs = self.combined_processor(text=txts, images=imgs, return_tensors="pt", padding=True,  truncation = True)
        outputs = self.model(**inputs)
        text_embeddings = outputs.text_embeds.detach().numpy()
        image_embeddings = outputs.image_embeds.detach().numpy()
        return (image_embeddings, text_embeddings)
        