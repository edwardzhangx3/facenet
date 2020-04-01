import os,cv2
import numpy as np
import scipy.stats as stats
import utils.textfile_utils as tutl
import utils.calculation_utils as cutl

"""
Give labels to face image embeddings.
"""

PREFIXES = [p + '_embeddings_detections' for p in ['whole_lab_training', 'abi_sandeep', 'csandeep_amine_andrew', 'james_apoorva_sandeep_1224']]
INPUT_EMB_PATHES = [os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'EMBEDDINGS', prefix + '.pkl') for prefix in PREFIXES]
SAMPLE_EMB_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'EMBEDDINGS', 'edge_embeddings.pickle')

OUTPUT_PKL_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'temp_labeled.pkl')
OUTPUT_PIC_DIR = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'face_pictures')

sample_emb = tutl.load_pkl(SAMPLE_EMB_PATH)

out_emb = {}

for path in INPUT_EMB_PATHES:
    embeddings = tutl.load_pkl(path)
    prefix = path.split(os.sep)[-1].split('.')[0].split('_embeddings_detections')[0]
    out_emb[prefix] = {}

    frame_num = 0
    while frame_num < len(embeddings):
        if len(embeddings[frame_num]) > 0:
            distances = np.array([cutl.distance(embeddings[frame_num][0]["embedding"], sample) for sample in sample_emb["embeddings"]])
            indexes = np.argsort(distances)
            top10labels = np.array([sample_emb["names"][i] for i in indexes[:10]])
            if np.mean(np.sort(distances)[:10]) < 0.2:
                majority = stats.mode(top10labels)[0][0]
            else:
                majority = 'unknown'
            embeddings[frame_num][0]["label"] = majority

            #Write jpg file.
            if not os.path.exists(os.path.join(OUTPUT_PIC_DIR, majority)): os.makedirs(os.path.join(OUTPUT_PIC_DIR, majority))
            cv2.imwrite(os.path.join(OUTPUT_PIC_DIR, majority, prefix + str(frame_num) + '.jpg'), embeddings[frame_num][0]["face_pixels"])
        out_emb[prefix][frame_num] = embeddings[frame_num]
        frame_num += 1

tutl.write_pkl(OUTPUT_PKL_PATH, out_emb)
