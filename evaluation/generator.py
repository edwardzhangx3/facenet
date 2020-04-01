import os,argparse
import utils.textfile_utils as tutl
import sklearn.utils

# DATASET generator for facenet embeddings

def setup_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data",
        help="path to data pkl")
    ap.add_argument("-n", "--data-size", type=int,
	    help="number of data for each label, balance option will be fixed True if specified")
    ap.add_argument("-l", "--label", nargs='+',
	    help="target label to pickle")
    ap.add_argument("-e", "--exclude-label", nargs='+',
	    help="exclude label to pickle")
    ap.add_argument("-o", "--output-file", required=True,
        help="path to output pkl")
    ap.add_argument("-s", "--shuffle", default=True,
        help="if False, data is not shuffled (Default: True)")
    ap.add_argument("-b", "--balanced", default=True,
        help="if False, all data will be saved to pickle, balance is ignored (Default: True)")
    return vars(ap.parse_args())

def gen_data(source=None, data_size=None, label=None, exclude_label=None, shuffle=True, balanced=True):

    if source is None:
        print("[INFO] loading default data source...")
        SOURCE_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', "dataset", 'learningset_all.pkl')
        source = tutl.load_pkl(SOURCE_PATH)

    KEYS = ['names', 'embeddings', 'video', 'frame']
    out = {k: [] for k in KEYS}

    if shuffle:
        print("[INFO] shuffling...")
        snames, sembeddings, svideo, sframe = sklearn.utils.shuffle(source['names'], source['embeddings'], source['video'], source['frame'])
        source = dict(names=snames, embeddings=sembeddings, video=svideo, frame=sframe)

    print("[INFO] filtering...")
    for i, name in enumerate(source['names']):
        if not label:
            if not exclude_label or not name in exclude_label:
                for k in KEYS: out[k].append(source[k][i])
        else:
            if name in label:
                for k in KEYS: out[k].append(source[k][i])

    labels = set(out['names'])

    if data_size or balanced:
        print("[INFO] balancing...")
        desired_size = data_size if data_size else float('inf')
        max_size = min({lbl: out['names'].count(lbl) for lbl in labels}.values())
        size = min(desired_size, max_size)
        #print('[DEBUG] size is ' + str(size))

        names, embeddings, video, frame = [], [], [], []

        idx = 0
        while min({lbl: names.count(lbl) for lbl in labels}.values()) < size:
            #print('[DEBUG] ' + str({label: names.count(label) for label in labels}))
            if names.count(out['names'][idx]) < size:
                names.append(out['names'][idx])
                embeddings.append(out['embeddings'][idx])
                video.append(out['video'][idx])
                frame.append(out['frame'][idx])
            idx += 1
    else:
        names, embeddings, video, frame = out['names'], out['embeddings'], out['vide'], out['frame']

    return dict(names=names, embeddings=embeddings, video=video, frame=frame)

def main():
    args = setup_args()
    if args['data']:
        print("[INFO] loading data source...")
        source = tutl.load_pkl(args['data'])
    else:
        source = None

    data = gen_data(source, args['data_size'], args['label'], args['exclude_label'], args['shuffle'], args['balanced'])

    print("[INFO] create pickle file...")
    tutl.write_pkl(args['output_file'], data)

if __name__ == '__main__':
    main()
