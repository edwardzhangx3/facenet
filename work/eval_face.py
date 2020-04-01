import os, argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utils.plotting_utils import *
import utils.textfile_utils as tutl

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", nargs='+', required=True,
	help="path to model trained")
ap.add_argument("-g", "--graph",
    help="path to graph of accuracy plot")
ap.add_argument("-t", "--testset", required=True,
    help="path to testset piclke")
args = vars(ap.parse_args())

print("[INFO] get testdata...")
testdata = tutl.load_pkl(os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', "dataset", 'testset.pkl'))
X, labelstr = [emb.reshape(128) for emb in testdata['embeddings']], testdata['names']

print("[INFO] loading model...")
models, les = [], []
for mpath in args["model"]:
    model_obj = tutl.load_pkl(mpath)
    if isinstance(model_obj, list):
        for model_obj_elem in model_obj:
            models.append(model_obj_elem[0])
            les.append(model_obj_elem[1])
    elif isinstance(model_obj, tuple):
        (model_elem, le_elem) = tutl.load_pkl(mpath)
        models.append(model_elem)
        les.append(le_elem)
    else:
        print("[WARN] Cannot parse model object. skip...")

print("[INFO] encoding labels...")
labels = [le.transform(labelstr) for le in les]

#predict and evaluate
print("[INFO] evaluating...")
y_preds = [model.predict(X) for model in models]
accuracy_set = []
target_accuracy_set, target_label = [], 'sandeep'
for le, label, y_pred in zip(les, labels, y_preds):
    print('[INFO] Confusion matrix...')
    print('[INFO] labels: ' + str(le.classes_))
    conf_mtx = confusion_matrix(label, y_pred)
    tidx = list(le.classes_).index(target_label)
    print(80/100)
    print(conf_mtx)
    accuracy_set.append(accuracy_score(label, y_pred))
    target_accuracy_set.append(float(conf_mtx[tidx][tidx])/float(sum(conf_mtx[tidx])))
print("[INFO] Accuracy set")
print(accuracy_set)
print(target_accuracy_set)

if args["graph"]:
    #generate graph
    # now plot a pdf of the distances between groups CONDITIONED ON A CERTAIN PERSON
    xlabel = 'Rounds#'
    ylabel = 'Accuracy'
    title_str = None
    plot_file = args['graph']
    #basic_plot_ts(ts_vector=accuracy_set, xlabel=xlabel, ylabel=ylabel, plot_file=plot_file, title_str=title_str, xticks=([0,4,9,14], [1,5,10,15]))
    overlaid_ts(normalized_ts_dict = dict(overall=dict(ts_vector=accuracy_set), target=dict(ts_vector=target_accuracy_set)), title_str = title_str, plot_file = plot_file, ylabel =ylabel, xlabel =xlabel, fontsize = 30, xticks = ([0,4,9,14], [1,5,10,15]), ylim = (0, 1.0), DEFAULT_ALPHA = 1.0, legend_present = True, DEFAULT_MARKERSIZE = 15, delete_yticks = False, xlim = None)
