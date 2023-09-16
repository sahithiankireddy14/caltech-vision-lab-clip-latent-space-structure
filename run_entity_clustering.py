import numpy as np
import pyarrow.parquet as pq 
import pandas as pd
import glob
import random
import requests
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
from datetime import datetime
from random import sample 

from itertools import cycle
from sklearn.metrics.pairwise import cosine_similarity
from clip_model import CLIP
from clip_prefix_captioning_inference import ClipCaptionModel, generate2, generate_beam
import clip as clip_github
import open_clip

from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import fowlkes_mallows_score, normalized_mutual_info_score, homogeneity_score

import skimage.io as io
import torch
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer
import PIL.Image




class Entity:
    def __init__(self, name, modality):
        self.name = name 
        self.embeddings = None
        self.count = 0
        self.modality = modality
        
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def set_count(self, count):
        self.count = count



clip = CLIP("openai/clip-vit-base-patch32")

class_subdirectories = {"Cat":["n02123045", "n02123159", "n02123394", "n02123597", "n02124075"], 
              "Dog": ["n02085936","n02097474", "n02105641",  "n02105855", "n02106030" ],
              "Plants":["n11939491", "n12057211"], "People":["n10148035", "n09835506"], 
              "Cars":["n02930766", "n03594945", "n04285008", "n03100240", "n03670208" ], 
              "Balls":["n02799071", "n02802426", "n03445777", "n04254680", "n04409515"], 
              "Accessories":["n04162706", "n02817516", "n02869837",  "n03124170", "n04259630"], 
              "Appliances":["n03761084", "n04442312", "n03207941", "n04070727", "n04517823" ], 
               "Fruit":["n07745940", "n07753592", "n07747607", "n07754684", "n07753275"],
               "Other Food":["n07697537", "n07697313", "n07873807", "n07614500",  "n07871810"]}



                

def main(images_dir, labels_csv, dataset_name, classes, preprocess_image_data, num_images, generate_texts):
    #perfectly_clustered_data(50, 100, 0.1,"zij-gaussian-0.1sigma-100classes-50perclass.pk1" )
    

    labels_mapping_dict = csv_to_dict(labels_csv)
    if preprocess_image_data:
        if dataset_name == "ImageNet":
            if generate_texts:
                #generate_coca_captions(images_dir, 600, 2)
                coca_caption_image_net(num_images)
            else:
                entities = process_imagenet(images_dir, num_images, generate_texts)
        elif dataset_name == "LAION":
            #process_laion(images_dir)
            test_laion(num_images)
    else:
        for directory in classes:
            paths = sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), num_images)
            if generate_texts:
                text_entity = Entity(labels_mapping_dict[directory][0], "text")
                text_entity.set_embeddings(clip.encode_text(caption(paths)))
                text_entity.set_count(num_images)
                entities.append(text_entity)
            else:
                images = []
                image_entity = Entity(labels_mapping_dict[directory][0], "image")
                for img in paths:
                    images.append(Image.open(img))
                image_entity.set_embeddings(clip.encode_image(images))
                image_entity.set_count(num_images)
                entities.append(image_entity)
    
    #compute_clustering_metrics_different_flavors(entities, len(entities), num_images)
    #cosine_sim_confusion_matrix(entities, num_images, len(entities))


    #overall_distribution_graph(images_dir, num_images)
    #class_distribution_graph(entities, images_dir, labels_mapping_dict, num_images)
    #overall_class_avg_cosine_overlay(images_dir, labels_mapping_dict, num_images)
    #intrisic_dimensionality(images_dir, labels_mapping_dict, num_images)

    #image_spectrum_graph(images_dir, labels_mapping_dict, 5, 100)
    #single_class_image_spectrum(images_dir, labels_mapping_dict, 100)

   




def compute_clustering_metrics_different_flavors(entities, num_classes, num_images):
    all_embeddings = entities[0].embeddings
    for i in range(1, len(entities)):
        all_embeddings = np.concatenate((all_embeddings, entities[i].embeddings), axis = 0)

    random_vectors = random_data(num_images, num_classes)

    
    with open("z-ij-guassian-0.1sigma.pk1", 'rb') as file:
        centroids = pickle.load(file)
        all_points_perfect_data = pickle.load(file)

    cosine_sim_no_pca = [np.round(spectral_clustering(all_embeddings, len(entities), entities, "cosine-sim")[2:], 3), None, None]
    cosine_sim_pca = [None, None, None]
    rbf_pca = [np.round(spectral_clustering(pca(all_embeddings, entities, 10), num_classes, entities, "rbf")[2:], 3), np.round(spectral_clustering(pca(random_vectors, entities, 10), num_classes, entities, "rbf")[2:], 3), np.round(spectral_clustering(pca(all_points_perfect_data, entities, 10), num_classes, entities, "rbf")[2:], 3)]


    row_headers = ["PCA + CosSim", "No PCA + CosSim", "PCA + RBF"]
    column_headers = ["x_ij", "y_ij", "z_ij"]

    data = [cosine_sim_no_pca, cosine_sim_pca, rbf_pca]
    fig, ax = plt.subplots(figsize=(10, 6))
    table = ax.table(cellText=data, colLabels=column_headers, rowLabels=row_headers,
                    cellLoc="center", loc="center")

  
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.3)  


    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(fontweight='bold')
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("clustering-metrics-diff-combos.png", format = 'png')
    plt.show()
 


   # TODO: pca with cosine sim affintiy doesnt' work for xij +  anything with cosine sim affintiy for y_ij or z_ij doesn't work
    # for x in [all_embeddings, random_vectors, all_points_perfect_data]:
    #      _, _, fm1, h1, nmi1 = spectral_clustering(x, len(entities), entities, "cosine-sim")
    #      print(fm1)
    #      _,_, fm2, h2, nmi2 = spectral_clustering(pca(x, entities, 10), num_classes, entities, "cosine-sim")
    #      print(fm2)
    #      _, _, fm3, h3, nmi3 = spectral_clustering(pca(x, entities, 10), num_classes, entities, "rbf")
    #      print(fm3)

        



def cosine_sim_confusion_matrix(entities, num_images, num_classes):
     all_embeddings = entities[0].embeddings
     for i in range(1, len(entities)):
        all_embeddings = np.concatenate((all_embeddings, entities[i].embeddings), axis = 0)
    
     labels_x = []
     for entity in entities:
         labels_x.append("")
         labels_x.append(entity.name)

     labels_yz = []
     for i in range(len(entities)):
         labels_yz.append("")
         labels_yz.append("Class " + str(i + 1 ))

     
     plot_heatmap(cosine_similarity(all_embeddings), labels_x, num_images, num_classes, "Cosine Similarity Between ImageNet Classes ", "cosine-sim-heatmap-imagenet-xij.png")
     plot_heatmap(cosine_similarity(random_data(num_images, num_classes)), labels_yz, num_images, num_classes, "Cosine Similarity Between Random Data ", "cosine-sim-heatmap-random-yij.png")
     plot_heatmap(cosine_similarity(perfectly_clustered_data(num_images, num_classes, 0.1, "z-ij-guassian-0.1sigma.pk1")), labels_yz, num_images, num_classes, "Cosine Similarity Between Perfectly Clustered Data ", "cosine-sim-heatmap-clustered-zij.png")
    
    
    
    
def random_data(num_images, num_classes):
    random = []
    for _ in range(num_images * num_classes):
        random_vector = np.random.uniform(-1, 1, 512)
        random.append(random_vector / np.linalg.norm(random_vector))
    return random

def distance_between_points(point1, point2):
    return np.linalg.norm(point1 - point2)

def normalize_array(data):
    norm = []
    for x in data:
        norm.append(x / np.linalg.norm(x))
    return np.array(norm)

def perfectly_clustered_data(num_images, num_classes, sigma, pickle_file_path):
    centroids = np.zeros((num_classes, 512))
    generated_centroids = 0

    while generated_centroids < num_classes:
        new_centroid =  np.random.uniform(-1, 1, 512)
        #new_centroid = new_centroid / np.linalg.norm(new_centroid)

        distance_to_existing_centroids = [distance_between_points(new_centroid, centroid) for centroid in centroids[:generated_centroids]]
       
        if all(distance > (3 * sigma) for distance in distance_to_existing_centroids):
            centroids[generated_centroids] = new_centroid
            generated_centroids += 1


    all_clusters = normalize_array(np.random.normal(loc=centroids[0], scale=sigma, size=(num_images, 512)))
    for i in range(1, len(centroids)):
        cluster_samples = np.random.normal(loc=centroids[i], scale=sigma, size=(num_images, 512))
        all_clusters = np.concatenate((all_clusters, normalize_array(cluster_samples)), axis = 0)
        

    # all_points = np.concatenate((normalize_array(np.random.normal(loc=centroids[0], scale=sigma, size=(num_images - 1, 512))), np.array([centroids[0]])), axis = 0)
    # for i in range(1, len(centroids)):
    #     cluster_samples = np.random.normal(loc=centroids[i], scale=sigma, size=(num_images - 1, 512))
    #     all_points = np.concatenate((all_points, np.concatenate((normalize_array(cluster_samples), np.array([centroids[i]])), axis = 0)), axis = 0)
        

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(centroids, file)
        pickle.dump(all_clusters, file)

    return all_clusters

def random_pairs_anchor_points_cosine_sim(data, num_anchor, num_images):
    total = []
    for _ in range(num_anchor):
        cosine_sim_per_anchor_pt = []
        anchor = random.choice(data)
        for _ in range(num_images):
            pt = random.choice(data)
            cosine_sim_per_anchor_pt.append(np.dot(anchor,pt))
        total.append(cosine_sim_per_anchor_pt)
    return total


def random_pairs_cosine_sim(data, num):
    cosine_sims = []
    for _ in range(num):
        pair = sample(data, 2)
        cosine_sims.append(np.dot(pair[0],pair[1]))
    return cosine_sims  


def pca(all_embeddings, entities, num_components):
    #TODO: Change to pass in coloring whether that's ground truth or predicted cluster labels

    if num_components != 2:
        pca_total = PCA(n_components=num_components).fit_transform(all_embeddings) 
    else:
        labels = [obj.name for obj in entities]
        ground_truth = get_ground_truth(entities)
        pca_total = PCA(n_components=2).fit_transform(all_embeddings) 
        visualize_scatter(1, pca_total[:, 0], pca_total[:, 1], ground_truth, True, labels, 
              "Principal Component 1", "Principal Component 2", "PCA", "plots/pca_" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")  + ".png")

    return pca_total



def spectral_clustering(values, n_clusters, entities, affinity): 
    if (affinity == "cosine-sim"):
         cosine_similarity_matrix = cosine_similarity(values)
         spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels="discretize", random_state=0)
         predicted_clusters = spectral_clustering.fit_predict(cosine_similarity_matrix)
    else:
        # Uses Default
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
        predicted_clusters = spectral_clustering.fit_predict(values)

    fm, h, nmi = evaluate(entities, predicted_clusters)

    return (predicted_clusters, spectral_clustering.affinity_matrix_, fm, h, nmi)

def tsne(entities):
    labels = [obj.name for obj in entities]
    ground_truth = get_ground_truth(entities)
    all_embeddings = entities[0].embeddings
    for i in range(1, len(entities)):
        all_embeddings = np.concatenate((all_embeddings, entities[i].embeddings), axis = 0)
    tsne_total = TSNE(n_components=2,  perplexity=3, random_state = 0).fit_transform(all_embeddings) 

    visualize_scatter(0, tsne_total[:, 0], tsne_total[:, 1], ground_truth, True, labels, 
              "", "",  "t-SNE", "plots/t-sne_" +  datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")  + ".png")
    
    return tsne_total

def evaluate(entities, predicted_clusters):
    ground_truth = get_ground_truth(entities)
    return (fowlkes_mallows_score(ground_truth, predicted_clusters),homogeneity_score(ground_truth, predicted_clusters), normalized_mutual_info_score(ground_truth, predicted_clusters))


def get_ground_truth(entities):
    ground_truth = []
    count = 0
    for entity in entities:
        ground_truth += entity.count * [count]
        count = count + 1
    return ground_truth

 

def process_laion(path):
    #laion_data = pd.read_csv(processed_csv) 
    laion_data = pq.read_table(path).to_pandas()
    pd.set_option('display.max_colwidth', None)
    laion_data = laion_data[:1500]
    images = []
    texts = []
    for img_url,text in zip(laion_data['URL'],list(laion_data['TEXT'])) :
        print(img_url)
        try:
            r = Image.open(requests.get(img_url, timeout=1, stream=True).raw)
        except:
            continue
        else:
            images.append(r)
            texts.append(text)
    print(len(texts))
    print(len(images))
    with open('laion-captions-images.pk1', 'wb') as file:
        pickle.dump(images, file)
        pickle.dump(texts, file)
   


def test_laion(num_samples):
     with open('laion-captions-images.pk1', 'rb') as file:
        images = pickle.load(file)
        texts = pickle.load(file)
     start_index = random.randint(0, len(images) - num_samples)
     image_embeddings, text_embeddings = clip.encode_both(images[start_index:start_index + num_samples], texts[start_index:start_index + num_samples])
     with open('laion-embeddings.pk1', 'wb') as file:
        pickle.dump(image_embeddings, file)
        pickle.dump(text_embeddings, file)
     create_images_text_heatmap(image_embeddings, text_embeddings, "LAION")
     intra_inter_modality_cosine_sim(image_embeddings, text_embeddings, num_samples, "LAION")


def generate_coca_captions(images_dir, num_classes, per_class):
     images = []
     paths = []
     for directory in sample(glob.glob(images_dir +  "/*"), num_classes):
        for image in sample(glob.glob(directory + "/*.JPEG"), per_class):
            paths.append(image)
            images.append(Image.open(image))
     captions = coca(paths)
     with open('image-net-coca-captioning-captions-images.pk1', 'wb') as file:
        pickle.dump(images, file)
        pickle.dump(captions, file)
     
     
def coca_caption_image_net(num_samples):
     with open('image-net-coca-captioning-captions-images.pk1', 'rb') as file:
       images =  pickle.load(file)
       captions =  pickle.load(file)

     image_embeddings, text_embeddings = clip.encode_both(images[:num_samples], captions[:num_samples])
     with open('image-net-coca-captioning-embeddings.pk1', 'wb') as file:
        pickle.dump(image_embeddings, file)
        pickle.dump(text_embeddings, file)

     create_images_text_heatmap(image_embeddings, text_embeddings, "ImageNet")
     intra_inter_modality_cosine_sim(image_embeddings, text_embeddings, num_samples, "ImageNet")


def coca(paths):
  model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
  captions = []
  c = 0
  for image_path in paths:
     im = Image.open(image_path).convert("RGB")
     im = transform(im).unsqueeze(0)
     generated = model.generate(im, num_beam_groups = 1)
     captions.append(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
     c = c + 1
     print(c)
  return captions



def process_imagenet(images_dir, num_images, generate_texts):
    # TODO: fix count 
    entity_objects = []
    for class_name in list(class_subdirectories.keys()):
        entity = Entity(class_name, "image")
        if generate_texts:
            entity.modality = "text"
        images = []
        texts = []
        for directory in class_subdirectories[class_name]:
            paths = sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), int(num_images / len(class_subdirectories[class_name])))
            if generate_texts:
                texts += caption(paths)
            else:
                for img in paths:
                    images.append(Image.open(img))
            
        if generate_texts:
             entity.set_embeddings(clip.encode_text(texts))
             entity.set_count(len(texts))

        else:
            entity.set_embeddings(clip.encode_image(images))
            entity.set_count(len(images))
        entity_objects.append(entity)
    
    return entity_objects   



def caption(paths):
    model_path = "conceptual_weights.pt"
    clip_github_model, preprocess = clip_github.load("ViT-B/32", jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    use_beam_search = False 
    captions = []
    for path in paths:
        image = io.imread(path)
        pil_image = PIL.Image.fromarray(image)
        image = preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            prefix = clip_github_model.encode_image(image)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            captions.append(generate_beam(model, tokenizer, embed=prefix_embed)[0])
        else:
            captions.append(generate2(model, tokenizer, embed=prefix_embed))

    return captions


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


# ------------------------------------------------------------------------------
# PLOTTING FUNCTIONS 
# ------------------------------------------------------------------------------

def intra_inter_modality_cosine_sim(image_embeddings, text_embeddings, num_samples, dataset_name):
     color_map = plt.cm.get_cmap('viridis')
     distinct_colors = [color_map(i) for i in np.linspace(0, 1, 3)]
     img_img_cosine_sims = []
     txt_txt_cosine_sims = []
     pairwise_cosine_sims = []
     for _ in range(num_samples):
        index1 = random.randint(0, len(image_embeddings) - 1)
        index2 = index1
        while index2 == index1:
            index2 = random.randint(0, len(image_embeddings) - 1)
        img_img_cosine_sims.append(np.dot(image_embeddings[index1],image_embeddings[index2])) 
        txt_txt_cosine_sims.append(np.dot(text_embeddings[index1],text_embeddings[index2])) 
        pairwise_cosine_sims.append(np.dot(image_embeddings[index1],text_embeddings[index1]))

     with open(dataset_name + '-intra-inter-modality.pk1', 'wb') as file:
        pickle.dump(img_img_cosine_sims, file)
        pickle.dump(txt_txt_cosine_sims, file)
        pickle.dump(pairwise_cosine_sims, file)

     plt.figure()
     plot_single_histogram_series(img_img_cosine_sims, 10, distinct_colors[0], 'solid', "Image-Image")
     plot_single_histogram_series(txt_txt_cosine_sims, 10, distinct_colors[1], 'solid', "Text-Text")
     plot_single_histogram_series(pairwise_cosine_sims, 10, distinct_colors[2], 'solid', "Image-Text Pairs")
     plt.xlim(0.0, 1.0)
     plt.legend()
     plt.xlabel('Cosine Similarity')
     plt.ylabel('Frequency')
     plt.title('Modality Cosine Similarity in ' + dataset_name )
     plt.savefig(dataset_name + "-intra-inter-modality-cosine-sim_500samples.png", format = 'png')



def create_images_text_heatmap(image_embeddings, text_embeddings, dataset_name):
   total = np.concatenate((image_embeddings, text_embeddings), axis = 0)
   cosine_matrix  = cosine_similarity(total, total)
   plt.figure(figsize=(10, 8))
   plt.imshow(cosine_matrix, cmap='viridis')
   
   labels = ["Images"]
   for i in range(len(cosine_matrix) - 1):
    if i == len(image_embeddings) - 1:
        labels.append("Texts")
    else:
        labels.append("")
    
   plt.xticks(np.arange(len(cosine_matrix)), labels=labels)
   plt.yticks(np.arange(len(cosine_matrix)), labels=labels)
   plt.colorbar()
   plt.title("Cosine Similarity Between " + dataset_name + " Image Text Pairs")
   plt.savefig('cosine-sim-image-text-' + dataset_name + "500samples.png", format = 'png')


def plot_heatmap(data, labels,  num_images, num_classes, title, pathname):
     plt.figure(figsize=(10, 8))
     plt.imshow(data, cmap='viridis')
     plt.xticks(range(1, (num_images * num_classes), int((num_images * num_classes) / (num_classes * 2))), [labels[int(i / int((num_images * num_classes) / (num_classes * 2)))] for i in range(1,  (num_images * num_classes) + 1, int((num_images * num_classes) / (num_classes * 2)))], rotation=45)
     plt.yticks(range(1, (num_images * num_classes), int((num_images * num_classes) / (num_classes * 2))), [labels[int(i / int((num_images * num_classes) / (num_classes * 2)))] for i in range(1,  (num_images * num_classes) , int((num_images * num_classes) / (num_classes * 2)))])
     plt.colorbar()
     plt.title(title)
     plt.savefig(pathname, format = 'png')


def zij_intraclass_data(num_classes, per_class):
    with open("zij-gaussian-0.1sigma-10classes-900perclass.pk1", 'rb') as file:
        centroids = pickle.load(file)
        z_ij = pickle.load(file)
    zij_intraclass_data = []
    for i in range(num_classes):
       zij_intraclass_data.append(z_ij[i * per_class:(i + 1) * per_class])
    return zij_intraclass_data

def zij_interclass_data():
    with open("zij-gaussian-0.1sigma-100classes-50perclass.pk1", 'rb') as file:
        centroids = pickle.load(file)
        z_ij = pickle.load(file)
    return z_ij

def xij_intraclass_data(images_dir, labels_mapping_dict, num_classes, per_class):
    xij_intraclass = []
    labels = []
    for directory in sample(glob.glob(images_dir +  "/*"), num_classes):
        imgs = []
        labels.append(str(labels_mapping_dict[directory.split("/")[-1]][0]))
        for image in sample(glob.glob(directory + "/*.JPEG"), per_class):
            imgs.append(Image.open(image))
        xij_intraclass.append(clip.encode_image(imgs))
    return xij_intraclass, labels

def xij_interclass_data(images_dir, num_classes, per_class):
    x_ij = []
    for directory in sample(glob.glob(images_dir +  "/*"), num_classes):
        for image in sample(glob.glob(directory + "/*.JPEG"), per_class):
            x_ij.append(clip.encode_image(Image.open(image))[0])
    return x_ij


def single_class_image_spectrum(images_dir,labels_mapping_dict, per_class):
     directory =  random.choice(glob.glob(images_dir +  "/*"))
     imgs = []
     paths = []
     for image in sample(glob.glob(directory + "/*.JPEG"), per_class):
        imgs.append(Image.open(image))
        paths.append(image)
     xij_intraclass = clip.encode_image(imgs)
     class_cosinesim_count = {}
     for _ in range(per_class * 5):
        index1 = random.randint(0, len(xij_intraclass) - 1)
        index2 = index1
        while index2 == index1:
            index2 = random.randint(0, len(xij_intraclass) - 1)
        cosine_sim = round(np.dot(xij_intraclass[index1], xij_intraclass[index2]), 1)
        if cosine_sim not in class_cosinesim_count:
            class_cosinesim_count[cosine_sim] = []
        class_cosinesim_count[cosine_sim].append((paths[index1], paths[index2]))
     fig, axes = plt.subplots(1, len(class_cosinesim_count) * 2, figsize=(15, 2), subplot_kw={'xticks': [], 'yticks': []})
     c = 0
     axes[0].set_ylabel(labels_mapping_dict[directory.split("/")[-1]][0].capitalize().replace(" ", "\n"), rotation=0, labelpad=30, va='center')
     for key in sorted(class_cosinesim_count.keys()):
         pair =  random.choice(class_cosinesim_count[key])
         axes.flat[c].imshow(plt.imread(pair[0]))
         axes.flat[c].set_title(str(key), loc = "left")
         axes.flat[c + 1].imshow(plt.imread(pair[1]))
         c = c + 2
    
     fig.suptitle("Varying Levels of Intraclass Cosine Similarity")
     plt.tight_layout()
     plt.savefig("single_class_image_spectrum.png", format = 'png')
        
         
     
def image_spectrum_graph(images_dir, labels_mapping_dict,  num_classes, per_class):
    xij_intraclass = []
    xij_paths = []
    num_cols = 4
    labels = []
    
    for directory in sample(glob.glob(images_dir +  "/*"), num_classes):
            imgs = []
            paths = []
            for image in sample(glob.glob(directory + "/*.JPEG"), per_class):
                imgs.append(Image.open(image))
                paths.append(image)
            xij_paths.append(paths)
            xij_intraclass.append(clip.encode_image(imgs))
            labels.append(labels_mapping_dict[directory.split("/")[-1]][0])
    fig, axes = plt.subplots(num_classes, num_cols, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(xij_intraclass)):
        med = []
        high = []
        for _ in range(per_class * 5):
            index1 = random.randint(0, len(xij_intraclass[i]) - 1)
            index2 = index1
            while index2 == index1:
                index2 = random.randint(0, len(xij_intraclass[i]) - 1)
            cosine_sim = round(np.dot(xij_intraclass[i][index1], xij_intraclass[i][index2]), 1)
            if cosine_sim <=0.7:
                med.append((xij_paths[i][index1], xij_paths[i][index2]))
            elif cosine_sim <= 1.0:
                high.append((xij_paths[i][index1], xij_paths[i][index2]))
    
        titles = ["Med: 0.4 - 0.7", "", "High: 0.8 - 1.0", ""]
        med_pair = random.choice(med)
        high_pair = random.choice(high)
        c = 0
        axes[i, 0].set_ylabel(labels[i].capitalize().replace(" ", "\n"), rotation=0, labelpad=30, va='center')
        for ax, title, img in zip(axes.flat[i * num_cols:(i + 1) * num_cols], titles, [med_pair[0], med_pair[1], high_pair[0], high_pair[1]]):
                ax.imshow(plt.imread(img))
                if c < num_cols and i == 0 and c % 2 == 0:
                     ax.set_title(title, loc = "left")
                c = c + 1
       

    fig.suptitle("Varying Levels of Intraclass Cosine Similarity")
    plt.tight_layout()
    plt.savefig("image_spectrum.png", format = 'png')
  
def plot_svd_error(sigmas, label):
    avg_sigma = np.mean(sigmas, axis=0)
    plt.figure()
    x = np.arange(len(avg_sigma) + 1)
    y = []
    for r in x:
      y.append(np.sqrt(sum((avg_sigma[i - 1]  * avg_sigma[i - 1])for i in range(r+1, len(avg_sigma) + 1))))
    plt.plot(x, y, 'b')
    plt.xlabel('R')
    plt.ylabel('e(r)')
    plt.title('Average ' + label + " Error in Choosing R Dimensions")
    plt.savefig(label + "_svd_error_r_dimensions.png", format = 'png')



# TODO: LOOK INTO BATCHING
def intrisic_dimensionality(images_dir, labels_mapping_dict, num_samples):
     num_classes = 5
     sigmas = []
     colors = ['b', 'g', 'r']

     plt.figure()
     plt.yscale('log')
     plt.xscale('log')

     #Intraclass 
     xij_intraclass, labels = xij_intraclass_data(images_dir, labels_mapping_dict, num_classes, num_samples)
     for i in range(len(xij_intraclass)):
           _, s, _ = np.linalg.svd(xij_intraclass[i])
           sigmas.append(s) 
           plt.plot(np.arange(len(s)), s, colors[i] , label = labels[i])
     plt.legend()
     plt.xlabel('Nth Singular Values')
     plt.ylabel('Singular Value')
     plt.title('Intraclass Singular Values')
     plt.savefig("singular_values_intraclass.png", format = 'png')

     # Error vs singular values graph
     # Calculate error by taking the sqrt(sum of the squares of the singular value left)
     # after choosing only r dimesnions so thus it's r + 1 to 512 (ensure to choose a number of samples > 512, as  dim(S) = (min(M, N), ))
     plot_svd_error(sigmas, "Intraclass")
     
    

def plot_single_histogram_series(data, num_bins, color, style, label):
        hist, bin_edges = np.histogram(data, bins=num_bins) 
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_midpoints, hist, c = color, linestyle = style, label = label)
        return bin_midpoints, hist


def plot_overall_class_avg(overall, intraclass_data, n, z, num_bins, color, label):
    average_function = np.mean(intraclass_data, axis=0)
    confidence_interval = z * np.std(intraclass_data, axis=0) / np.sqrt(n)
    
    plot_single_histogram_series(overall, 10, color, "solid", label + " Interclass")
    plot_single_histogram_series(average_function, 10, color, "dashed",  label + " Average Intraclass")

    lb_ci_hist, lb_ci_bin_edges = np.histogram(average_function - confidence_interval, bins=num_bins) 
    ub_ci_hist, ub_ci_bin_edges = np.histogram(average_function + confidence_interval, bins=num_bins) 
    lb_ci_bin_midpoints = (lb_ci_bin_edges[:-1] + lb_ci_bin_edges[1:]) / 2
    ub_ci_bin_midpoints = (ub_ci_bin_edges[:-1] + ub_ci_bin_edges[1:]) / 2
    avg_ci_x = []
    for i in range(10):
        avg_ci_x.append((lb_ci_bin_midpoints[i] + ub_ci_bin_midpoints[i]) / 2)
    plt.fill_between(avg_ci_x, lb_ci_hist, ub_ci_hist, alpha=0.2)


def overall_class_avg_cosine_overlay(images_dir,labels_mapping_dict, num_samples):
    num_classes = 10
    per_class = 900

    plt.figure()
    xij_intraclass_cosine_sims = []
    xij_intraclass, _ = xij_intraclass_data(images_dir, labels_mapping_dict, num_classes, per_class)
    for x in xij_intraclass:
        xij_intraclass_cosine_sims.append(random_pairs_cosine_sim(x.tolist(), num_samples))
    
    xij_interclass_cosine_sims = random_pairs_cosine_sim(xij_interclass_data(images_dir, num_classes, per_class), num_samples)
    
    zij_intraclass_cosine_sims = []
    zij_intraclass = zij_intraclass_data(num_classes, per_class)
    for x in zij_intraclass:
        zij_intraclass_cosine_sims.append(random_pairs_cosine_sim(x.tolist(), num_samples))

    zij_interclass_cosine_sims = random_pairs_cosine_sim(zij_interclass_data().tolist(), num_samples)

    plot_overall_class_avg(xij_interclass_cosine_sims, xij_intraclass_cosine_sims, num_samples, 2.576, 10, 'b', "ImageNet")
    plot_overall_class_avg(zij_interclass_cosine_sims, zij_intraclass_cosine_sims, num_samples, 2.576, 10, 'g', "Perfectly Clustered")


    plt.legend()
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Cosine Similarity Within and Across Classes')
    plt.savefig("cosine_sim_interclass_intraclass_normal_both.png", format = 'png')
    plt.xlim(0.8, 1.0)
    plt.savefig("cosine_sim_interclass_intraclass_log_both.png", format = 'png')

    
   


def class_distribution_graph(entities, images_dir, labels_mapping_dict, num_samples):
     num_classes = 10
     per_class = 900
     num_bins = 10
     color_map = plt.cm.get_cmap('viridis')
     distinct_colors = [color_map(i) for i in np.linspace(0, 1, num_classes)]
    
    
     # Xij Intraclass
    #  plt.figure()
    #  xij_intraclass, labels = xij_intraclass_data(images_dir, labels_mapping_dict, num_classes, per_class)
    #  for i in range(len(xij_intraclass)):
    #     x_ij_class_cosine_similarities = random_pairs_cosine_sim(xij_intraclass[i].tolist(), num_samples)
    #     plot_single_histogram_series(x_ij_class_cosine_similarities, num_bins, distinct_colors[i], "solid", labels[i])
    #  plt.legend()
    #  plt.xlim(0.0, 1.0)
    #  plt.xlabel('Cosine Similarity')
    #  plt.ylabel('Frequency')
    #  plt.title('Cosine Similarity Within ImageNet Classes')
    #  plt.savefig("xij_cosine_sim_normal.png", format = 'png')
    #  plt.xlim(0.8, 1.0)
    #  plt.savefig("xij_cosine_sim_tail.png", format = 'png')

     
     # Zij Intraclass
     plt.figure()
     zij_intraclass = zij_intraclass_data(num_classes, per_class)
     for i in range(len(zij_intraclass)):
        z_ij_class_cosine_similarities = random_pairs_cosine_sim(zij_intraclass[i].tolist(), num_samples)
        plot_single_histogram_series(z_ij_class_cosine_similarities, num_bins, distinct_colors[i], "solid", "Class " + str(i + 1))
     plt.legend()
     plt.xlim(0.4, 1.0)
     plt.xlabel('Cosine Similarity')
     plt.ylabel('Frequency')
     plt.title('Cosine Similarity Within Perfectly Clustered Classes')
     plt.savefig("zij_cosine_sim_normal.png", format = 'png')
     plt.xlim(0.8, 1.0)
     plt.savefig("zij_cosine_sim_tail.png", format = 'png')

    # Xij Predetermined Classes
    #  plt.figure()
    #  for entity in entities:
    #     x_ij_class_cosine_similarities = random_pairs_cosine_sim(entity.embeddings.tolist(), num_samples)
    #     plot_single_histogram_series(x_ij_class_cosine_similarities, num_bins, distinct_colors[c], "solid", entity.name)
    #     c = c + 1
    #  plt.legend()
    #  plt.xlabel('Cosine Similarity')
    #  plt.ylabel('Frequency')
    #  plt.title('Cosine Similarity Within Predetermined ImageNet Classes')
    #  plt.savefig("xij_predetermined_cosine_sim_normal.png", format = 'png')
    #  plt.xlim(0.8, 1.0)
    #  plt.savefig("xij_predetermined_cosine_sim_tail.png", format = 'png')



def overall_distribution_graph(images_dir, num_samples):
    num_classes = 100
    per_class = 50
    num_bins = 10
    labels = ["x_ij", "y_ij", "z_ij"]
    color_map = plt.cm.get_cmap('viridis')
    colors = ['b', 'g', 'r']
    distinct_colors = [color_map(k) for k in np.linspace(0, 1, 4)]

    # Interclass
    x_ij = xij_interclass_data(images_dir, num_classes, per_class)
    y_ij = random_data(per_class, num_classes)
    z_ij = zij_interclass_data()
    

     # OVERALL RANDOM
    random_pairs = [random_pairs_cosine_sim(x_ij, num_samples), random_pairs_cosine_sim(y_ij, num_samples), random_pairs_cosine_sim(z_ij.tolist(), num_samples)]
    plt.figure(1)
    for i in range(len(random_pairs)):
        if i == 2:
            print(max(random_pairs[i]))
            plot_single_histogram_series(random_pairs[i], 20, distinct_colors[i + 1], "solid", labels[i])
        else:
            plot_single_histogram_series(random_pairs[i], num_bins, distinct_colors[i + 1], "solid", labels[i])
    plt.legend()
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Cosine Similarity Between Random Pairs')
    plt.savefig("overall_cosine_sim_normal.png", format = 'png')
    plt.title('Cosine Similarity Between Random Pairs in Log Scale')
    plt.yscale('log')
    plt.savefig("overall_cosine_sim_log.png", format = 'png')

    # OVERALL RANDOM WITH ANCHOR POINTS
    # random_anchor_pairs = [random_pairs_anchor_points_cosine_sim(x_ij, 10, num_samples), random_pairs_anchor_points_cosine_sim(y_ij, 10, num_samples), random_pairs_anchor_points_cosine_sim(z_ij.tolist(), 10, num_samples)]
    # for i in range(len(random_anchor_pairs)):
    #     plt.figure()
    #     for j in range(len(random_anchor_pairs[i])):
    #         plot_single_histogram_series(random_anchor_pairs[i][j], num_bins, distinct_colors[j], "solid", "Anchor Point " + str(j+1))
    #     plt.legend()
    #     plt.xlabel('Cosine Similarity')
    #     plt.ylabel('Frequency')
    #     plt.title('Cosine Similarity Between ' + labels[i] + ' Anchor Points')
    #     plt.savefig("overall_" + labels[i] + "_cosine_sim_anchor_pts_normal.png", format = 'png')
    

    
############# OLDER GRAPHS ##############

def create_mulitclass_image_text_graph(images_dir, num_images):
    entity_objects = []
    for class_name in list(class_subdirectories.keys()):
        entity = Entity(class_name, "both")
        images = []
        texts = []
        for directory in class_subdirectories[class_name]:
            paths = sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), int(num_images / len(class_subdirectories[class_name])))
            texts += caption(paths)
           
            for img in paths:
                    images.append(Image.open(img))
            
        image_embeddings, text_embeddings = clip.encode_image(images), clip.encode_text(texts)
        all_embeddings = np.concatenate((image_embeddings, text_embeddings), axis = 0)

        entity.set_embeddings(all_embeddings)
        entity.set_count(len(images) + len(texts))
        print(entity.count)
        entity_objects.append(entity)

    all_embeddings = entity_objects[0].embeddings
    for i in range(1, len(entity_objects)):
        all_embeddings = np.concatenate((all_embeddings, entity_objects[i].embeddings), axis = 0)
    pca_total = pca(all_embeddings, entity_objects, 2)
    
   
    
    labels = [obj.name for obj in entity_objects]
    plt.figure(10)
    for i in range(10):
        start_idx = i * num_images
        end_idx = (i + 1) * num_images
        plt.scatter(pca_total[:, 0][start_idx:start_idx + int((num_images / 2))], pca_total[:, 1][start_idx:start_idx + int((num_images / 2))], c=f'C{i}',
                    marker='*',  label = labels[i])

        plt.scatter(pca_total[:, 0][start_idx + int((num_images / 2)):end_idx], pca_total[:, 1][start_idx + int((num_images / 2)):end_idx], c=f'C{i}',
                    marker='o')
    

    plt.legend()
    plt.title("PCA of Encoded Image Text Pairs for " + str(10) + " Classes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 1")
    plt.savefig("pca_image_text_pairs_" + str(10) + ".png", format = "png")
     


def create_single_class_image_text_graph(images_dir, num_images, class_name):
     texts = []
     images = []
     for directory in class_subdirectories[class_name]:
          paths = sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), int(num_images / len(class_subdirectories[class_name])))
          texts += caption(paths)
          for img in paths:
                    images.append(Image.open(img))
     image_embeddings, text_embeddings = clip.encode_image(images), clip.encode_text(texts)
     all_embeddings = np.concatenate((image_embeddings, text_embeddings), axis = 0)
     all_embeddings = np.concatenate((all_embeddings, clip.encode_text("a photo of a " + class_name)), axis = 0)
     pca_total = PCA(n_components=2).fit_transform(all_embeddings) 
      
     
     plt.figure(6)
     plt.scatter(pca_total[:, 0][:10], pca_total[:, 1][:10], marker = "*" , c = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9], label='Image', cmap='viridis')
     plt.scatter(pca_total[:, 0][10:20], pca_total[:, 1][10:20], marker = "o" , c = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9], label='Text', cmap='viridis')
     plt.scatter(pca_total[:, 0][20:], pca_total[:, 1][20:],  marker = "d" , c = "black", label='Ground Truth Text', cmap='viridis')
     plt.legend()

     plt.title("PCA of Encoded Image Text Pairs for " + class_name + " Class")
     plt.xlabel("Principal Component 1")
     plt.ylabel("Principal Component 1")
     plt.savefig("pca_image_text_pairs_" + class_name + ".png", format = "png")
     



def create_fm_h_graph(images_dir, num_images):
    x = [2,3,4,5,6,7]
    multiple_seeds_h = []
    multiple_seeds_fm = []
    for seed in range(3):
        y_1 = []
        y_2 = []
        random.seed(seed)
        for num in range(2, 8):
            entity_objects = []
            for class_name in sample(list(class_subdirectories.keys()), num):
                entity = Entity(class_name, "image")
                images = []
                for directory in class_subdirectories[class_name]:
                    paths = sample(glob.glob(images_dir + "/"+ directory + "/*.JPEG"), int(num_images / len(class_subdirectories[class_name])))
                    for img in paths:
                        images.append(Image.open(img))
                    entity.set_embeddings(clip.encode_image(images))
                    entity.set_count(len(images))
                entity_objects.append(entity)

            all_embeddings = entity_objects[0].embeddings
            for i in range(1, len(entity_objects)):
                 all_embeddings = np.concatenate((all_embeddings, entity_objects[i].embeddings), axis = 0)
            _, _, fm, h, _ = spectral_clustering(pca(all_embeddings, entity_objects, 2), len(entity_objects), entity_objects, "")
            y_1.append(fm)
            y_2.append(h)
        multiple_seeds_fm.append(y_1)
        multiple_seeds_h.append(y_2)
   

    plt.figure(4)
    for i in range(len(multiple_seeds_fm)):
        plt.plot(x, multiple_seeds_fm[i], label = "Permuation " + str(i + 1))
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.title("Fowlkes Mallows Score from Clustering of a Various Number of Classes")
    plt.xlabel("Number of Classes")
    plt.ylabel("Fowlkes Mallows Score")
    plt.savefig("multiple_image_classes_fowlkes_mallows.png", format = "png")
   
    plt.figure(5)
    for i in range(len(multiple_seeds_fm)):
        plt.plot(x, multiple_seeds_h[i], label = "Permuation " + str(i + 1))
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.title("Homogeneity Score from Clustering of a Various Number of Classes")
    plt.xlabel("Number of Classes")
    plt.ylabel("Homogeneity Score")
    plt.savefig("multiple_image_classes_homogeneity.png", format = "png")
       
    

def visualize_scatter(figure_count, x ,y, colors,legend, labels, xaxis_label, yaxis_label, title, figure_path):
    plt.figure(figure_count)
    scatter = plt.scatter(x, y, s = 10, c = colors, cmap='viridis')
    if legend:
        plt.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")
    plt.title(title)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.savefig(figure_path, format='png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type = str, help="path to directory of image data")
    parser.add_argument("labels_csv", type = str, help="path to labels mapping text file")
    parser.add_argument("dataset_name", type = str, help="Specify one of the following dataset names: ImageNet, COCO")
    parser.add_argument( "classes", nargs="*", type=str, default=[],help="the different classes in the dataset to encode assumed to be subdirectory names in images-dir else set preprocess-image-data flag")
    parser.add_argument("--preprocess_image_data", action="store_true", help="set flag if the desired class names don't match the dataset labels")
    parser.add_argument("num_images", type = int, help="number of images to pull from each class / subdirectory")

    
    parser.add_argument("--generate_texts", action="store_true", help="set flag to generate captions for images and enocde these texts instead")

    

   

    args = parser.parse_args()
    main(**vars(args))