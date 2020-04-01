# per-person, compare inter and intra-group embedding L2 distances

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import sys, os
import numpy as np

ROOT_DIR = os.environ['HARVESTNET_ROOT_DIR']

utils_dir = ROOT_DIR + '/utils/'
sys.path.append(utils_dir)
from plotting_utils import *

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
        help="path to serialized db of facial embeddings")
ap.add_argument("-b", "--base-plot-dir", required=True)
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print('labels are: ', labels, data["names"])

# data['embeddings'] has the 128-d embedding, labels has the list of labels

name_to_embedding_dict = {}

for name, embedding in zip(data['names'], data['embeddings']):
    if name in name_to_embedding_dict.keys():
        name_to_embedding_dict[name].append(embedding)
    else:
        name_to_embedding_dict[name] = [embedding]

NUM_SAMPLES = 5

within_class_embedding_distance_list = []

outside_class_embedding_distance_list = []

for name, emb_list in name_to_embedding_dict.iteritems():

    # inter-class distances
    emb_distance_list = []
    for emb_x in emb_list:
        for emb_y in emb_list:
            dist = distance(emb_x, emb_y)
            emb_distance_list.append(dist)
            within_class_embedding_distance_list.append(dist)
    median_distance = np.median(emb_distance_list)

    print(' ')
    print('name: ', name)
    print('median_distance: ', median_distance)

    other_emb_distance_list = []

    # outside class distances
    for other_name, other_emb_list in name_to_embedding_dict.iteritems():
        if name != other_name:
            # for N times, sample one of our embeddings and one outside and find distance
            for num_sample in range(NUM_SAMPLES):
                index_curr_name = np.random.choice(range(len(emb_list))) 
                index_other_name = np.random.choice(range(len(other_emb_list)))

                emb_x = emb_list[index_curr_name]
                emb_y = other_emb_list[index_other_name]

                other_dist = distance(emb_x, emb_y)
                other_emb_distance_list.append(other_dist)
                outside_class_embedding_distance_list.append(other_dist)

    print('OTHER median_distance: ', np.median(other_emb_distance_list))
    print(' ')

    # now plot a pdf of the distances between groups CONDITIONED ON A CERTAIN PERSON
    data_vector_list = [emb_distance_list, other_emb_distance_list]
    xlabel = 'FaceNet Embedding Distance'
    title_str = name
    legend = ['within class', 'outside class']
    plot_file = args['base_plot_dir'] + '/' + str(name) + '_facenet.pdf'
    plot_several_pdf(data_vector_list = data_vector_list, xlabel = xlabel, plot_file = plot_file, title_str = title_str, legend = legend, norm = True)



# now plot a pdf of the distances between groups CONDITIONED ON A CERTAIN PERSON
data_vector_list = [within_class_embedding_distance_list, outside_class_embedding_distance_list]
xlabel = 'FaceNet Embedding Distance'
title_str = None
legend = ['within class', 'outside class']
plot_file = args['base_plot_dir'] + '/' + 'ALL_CLASS' + '_facenet.pdf'
plot_several_pdf(data_vector_list = data_vector_list, xlabel = xlabel, plot_file = plot_file, title_str = title_str, legend = legend, norm = True)

