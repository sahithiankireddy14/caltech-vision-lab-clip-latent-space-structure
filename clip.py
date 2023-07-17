
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor


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

    def encode_image(self, images):
        inputs = self.image_processor(images, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs).detach().numpy()
        return image_features
         
    def encode_text(self, texts):
         inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
         text_features = self.model.get_text_features(**inputs).detach().numpy()
         return text_features
        