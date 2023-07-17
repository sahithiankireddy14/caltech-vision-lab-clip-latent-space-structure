from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from random import sample 
from clip import CLIP


clip = CLIP("openai/clip-vit-base-patch32")

class ImageEntity:
    def __init__(self, name):
        self.name = name 
        self.embeddings = None
    

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
    

def main(images_dir, labels_csv, dataset_name, classes, preprocess_image_data, num_images, texts=None):
    labels_mapping_dict = csv_to_dict(labels_csv)
    image_entity_objects = []
    
    if preprocess_image_data:
        if dataset_name == "ImageNet":
            image_entity_objects = process_imagenet(images_dir, num_images)
    else:
        for directory in classes:
            images = []
            image_entity= ImageEntity(labels_mapping_dict[directory][0])
            for img in sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), num_images):
                 images.append(Image.open(img))
            image_entity.set_embeddings(clip.encode_image(images))
            image_entity_objects.append(image_entity)

    spectral_clustering(pca(image_entity_objects), len(image_entity_objects))



def pca(image_entity_objects):
    pca_total = PCA(n_components=2).fit_transform(image_entity_objects[0].embeddings)
    for i in range(1, len(image_entity_objects)):
        pca_total = np.concatenate((pca_total, PCA(n_components=2).fit_transform(image_entity_objects[i].embeddings)), axis = 0)

def spectral_clustering(values, n_clusters):
    # TODO
    return




    
def process_imagenet(images_dir, num_images):
    image_entity_objects = []
    class_subdirectories = {"Cat":["n02123045", "n02123159", "n02123394", "n02123597", "n02124075"], 
              "Dog": ["n02085936","n02097474", "n02105641",  "n02105855", "n02106030" ],
              "Plants":["n11939491", "n12057211"], "People":["n10148035", "n09835506", "n10565667" ], 
              "Cars":["n02930766", "n03594945", "n04285008", "n03100240", "n03670208" ], 
              "Balls":["n02799071", "n02802426", "n03445777", "n04254680", "n04409515"], 
              "Accessories":["n04162706", "n02817516", "n02869837",  "n03124170", "n04259630"], 
              "Appliances":["n03761084", "n04442312", "n03207941", "n04070727", "n04517823" ], 
               "Fruit":["n07745940", "n07753592", "n07747607", "n07754684", "n07753275"],
               "Other Food":["n07697537", "n07697313", "n07873807", "n07614500",  "n07871810"]}
    
    for class_name in list(class_subdirectories.keys()):
        image_entity = ImageEntity(class_name)
        images = []
        for directory in class_subdirectories[class_name]:
            for img in sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), int(num_images / len(class_subdirectories[class_name]))):
                    images.append(Image.open(img))
        image_entity.set_embeddings(clip.encode_image(images))
        image_entity_objects.append(image_entity)
    return image_entity_objects            



def csv_to_dict(csv_file):
    labels_dict = {}
    with open(csv_file, 'r') as file:
        csv_data = file.read()
        rows = csv_data.split('\n')
        for row in rows:
            split = row.split(" ", 1)
            if len(split) < 2:
                continue
            labels_dict[split[0]] = split[1].split(",")
    return labels_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type = str, help="path to directory of image data")
    parser.add_argument("labels_csv", type = str, help="path to labels mapping text file")
    parser.add_argument("dataset_name", type = str, help="Specify one of the following dataset names: ImageNet, COCO")
    parser.add_argument( "classes", nargs="*", type=str, default=[],help="the different classes in the dataset to encode assumed to be subdirectory names in images-dir else set preprocess-image-data flag")
    parser.add_argument("--preprocess_image_data", action="store_true", help="set flag if the desired class names don't match the dataset labels")
    parser.add_argument("num_images", type = int, help="number of images to pull from each class / subdirectory")
    

    parser.add_argument( "--texts", nargs="*", type=str, default=[], help="texts to encode")
   

    args = parser.parse_args()
    main(**vars(args))