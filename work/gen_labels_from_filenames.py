import os
import utils.textfile_utils as tutl

"""
Generate learning data from locations and file names of picture files.
"""

PREFIXES = ['whole_lab_training', 'abi_sandeep', 'csandeep_amine_andrew', 'james_apoorva_sandeep_1224']
SUFFIX = '.jpg'
TEMP_PKL_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'temp_labeled.pkl')
IMAGE_FILE_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'face_pictures', 'ground_truth')
LABELS = [label for label in os.listdir(IMAGE_FILE_PATH) if label[0] != '.']
OUTPUT_PKL_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'face_learning_data.pkl')

temp_emb = tutl.load_pkl(TEMP_PKL_PATH)

out_emb = dict(names=[], embeddings=[], video=[], frame=[])

def get_video_file_name_and_frame_num(fname):
    for prefix in PREFIXES:
        if fname.find(prefix) != -1:
            return (prefix, int(fname.split(prefix)[1].split(SUFFIX)[0]))
    raise Exception('[ERR] No video file name found! ' + str(fname))

for label in LABELS:
    for vname_frame in [get_video_file_name_and_frame_num(fname) for fname in os.listdir(os.path.join(IMAGE_FILE_PATH, label)) if fname.find(SUFFIX) > 0]:
        video_name, frame = vname_frame[0], vname_frame[1]
        out_emb['names'].append(label)
        out_emb['embeddings'].append(temp_emb[video_name][frame][0]['embedding'])
        out_emb['video'].append(video_name)
        out_emb['frame'].append(frame)

tutl.write_pkl(OUTPUT_PKL_PATH, out_emb)
