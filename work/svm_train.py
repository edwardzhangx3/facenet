import os, argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import shuffle
import utils.textfile_utils as tutl

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings",
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer",
	help="path to output model trained to recognize faces")
ap.add_argument("-n", "--data_size",
    help="number of embeddings per label for learning")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
LEARNING_DATA_PKL_PATH = args['embeddings'] if args['embeddings'] else os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', "dataset", 'face_learning_data.pkl')
ldata = tutl.load_pkl(LEARNING_DATA_PKL_PATH)
orig_embeddings, label_str = np.array([embedding.reshape(128) for embedding in ldata['embeddings']]), ldata['names']

LABEL_LIST = set(label_str)

print("[INFO] shuffle data...")
emb_dict = {label: shuffle([embedding for i, embedding in enumerate(orig_embeddings) if label == label_str[i]]) for label in LABEL_LIST}

print("[INFO] select embeddings...")
num = int(args['data_size']) if args['data_size'] else np.min([len(emb) for label, emb in emb_dict.items()])
embeddings = []
for label in LABEL_LIST: embeddings.extend(emb_dict[label][:num])

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(np.array([[label for i in range(num)] for label in LABEL_LIST]).flatten())

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(embeddings, labels)

print("[INFO] save model...")
tutl.write_pkl(args["recognizer"] if args["recognizer"] else os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'models', 'face_recg_' + str(num) + '.pkl'), (recognizer, le))
