import os,argparse,cv2
import utils.textfile_utils as tutl

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data",
	help="path to serialized db of facial embeddings")
ap.add_argument("-t", "--target", required=True,
	help="path to target data pkl to visualize")
ap.add_argument("-o", "--output",
    help="directory path to output images")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
DB_PATH = args['data'] if args['data'] else os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', "dataset", 'face_db.pkl')
db = tutl.load_pkl(DB_PATH)

target = tutl.load_pkl(args["target"])
OUT_DIR = args['output'] if args['output'] else os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'images')

print("[INFO] generate images...")
for name, emb, video, frame in zip(target['names'], target['embeddings'], target['video'], target['frame']):
    try:
        face_pixel = db[video][frame][0]['face_pixels']
        cv2.imwrite(os.path.join(OUT_DIR, video + '_' + str(frame) + '.jpg'), face_pixel)
    except Exception as inst:
        print("[ERR] Image search failed!")
        print(str(inst))

print("[INFO] Image search finished.")
