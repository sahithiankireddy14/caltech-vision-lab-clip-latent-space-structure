from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from transformers import CLIPProcessor, CLIPModel


'''
Initalizes instance of CLIP model and determines embeddings. 
Model is a ViT-B/32 Transformer architecture for the image encoder and a 
masked self-attention Transformer for the text encoder, all trained on data 
from various websites and commonly-used pre-existing image datasets such as 
YFCC100M. Takes in a array of images and texts in order to return the 
image and text emebeddings for the inputs.
'''

class CLIP:
  
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embeddings(self, imgs, txts):
        inputs = self.processor(text=txts, images=imgs, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # softmax to get the label probabilities
        image_embeddings = outputs.image_embeds.detach().numpy()
        text_embeddings = outputs.text_embeds.detach().numpy()
        for i in range(len(image_embeddings)):
                image_embeddings[i] = np.reshape(image_embeddings[i], (1, 512))
        for i in range(len(text_embeddings)):
                text_embeddings[i] = np.reshape(text_embeddings[i], (1, 512))
        return (image_embeddings, text_embeddings)



# CLIP needs atleast one image and one text in order to create this joint modality 
# space. As per our experiement, we aim to mainly deal with one modality? 

clip = CLIP()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["a photo of a cat"]

image_embeddings, text_embeddings = clip.embeddings(image, text)

