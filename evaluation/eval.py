import os,cv2,multiprocessing,copy
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from utils.plotting_utils import *
from sampler.sampler import FirstSampler
from sampler.sampler import TargetExtractor
from sampler.trainer import FaceNetEvaluator
from facenet.evaluation.generator import gen_data
import utils.textfile_utils as tutl
import utils.calculation_utils as cutl

DATA_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'dataset')
RESULT_OUT_PATH = os.path.join(os.environ["HARVESTNET_ROOT_DIR"], 'tmp', 'results')

#LEARNING DATA
ALL_DATA = os.path.join(DATA_PATH, 'face_learning_data.pkl') #This includes testset.
ALL_LEARNING_DATA = os.path.join(DATA_PATH, 'learningset_all.pkl')
LEARNING_DATA = os.path.join(DATA_PATH, 'learningset.pkl') #Balanced learning dataset that is ramdomly sampled.
ldata = tutl.load_pkl(LEARNING_DATA)

DATA0_SANDEEP = os.path.join(DATA_PATH, 'learningset_w_o_sandeep.pkl') #learning data except sandeep
DATA0_JAMES = os.path.join(DATA_PATH, 'learningset_w_o_james.pkl')
DATA0_2TARGETs = os.path.join(DATA_PATH, 'd0_2targets.pkl') #d0 data except targets

LEARNING_DATA_SANDEEP = os.path.join(DATA_PATH, 'ldata_d0_U_target.pkl')
LEARNING_DATA_JAMES = os.path.join(DATA_PATH, 'ldata_d0_U_james.pkl')
LEARNING_DATA_BOTH = os.path.join(DATA_PATH, 'ldata_d0_U_2targets.pkl')

ORACLE_SANDEEP = os.path.join(DATA_PATH, 'oracle_sandeep.pkl') #contains sandeep embeddings ONLY
ORACLE_JAMES = os.path.join(DATA_PATH, 'oracle_james.pkl') #contains james embeddings ONLY

empty = dict(names=[], embeddings=[], video=[], frame=[])


#TEST DATA
testset = tutl.load_pkl(os.path.join(DATA_PATH, 'testset.pkl'))
evaluator = FaceNetEvaluator(testset)

#TARGET DATA
TARGET_RSS = os.path.join(DATA_PATH, 'target_sandeep_RSS2019.pkl')
TARGET_RANDOM = os.path.join(DATA_PATH, 'target_sandeep_00.pkl')
TARGET_MANUAL = os.path.join(DATA_PATH, 'mtarget.pkl') #Sandeep

TARGET_RANDOM_JAMES = os.path.join(DATA_PATH, 'target_james_rand.pkl')
TARGET_MANUAL_JAMES = os.path.join(DATA_PATH, 'target_james.pkl')

#TODO collect larger target dataset

#default graph params
xlabel, ylabel = 'Round#', 'Accuracy'
xticks = ([0,4,9,14,19], [1,5,10,15,20])
ylim = (0, 1.0)

num_cpus = min(multiprocessing.cpu_count(), 32)

#### Common process ####

def train_and_evaluate(target, d0, learningset, cache_size, threshold, cache_method, weight, extr):
    sampler = FirstSampler(cache_size=cache_size, threshold=threshold, cache_method=cache_method)
    sampler.set_target(target, weight)
    sampler.trainer.set_training_data(d0['embeddings'], d0['names'])

    if 'pixels' in learningset.keys():
        extr.multiprocess = False
        extr.dataset = learningset['pixels']
        mat = [[extr.dist_matrix[learningset['orig'][i]][learningset['orig'][j]] for i in range(extr.dlength)] for j in range(extr.dlength)]
        extr.dist_matrix = mat
        vars = [extr.vars[learningset['orig'][i]] for i in range(extr.dlength)]
        extr.vars = vars
    else:
        extr = None

    frame_idx = 0
    result = []
    while frame_idx != -1 and sampler.retrain_count < FirstSampler._MAX_RETRAIN:
        frame_idx = sampler.process_once(learningset, frame_idx)
        recognizer, label_encoder = sampler.retrain()
        res = evaluator.eval(recognizer, label_encoder)
        sampler.updata_weight(res)
        result.append(res)

        if extr: #Target dynamic update.
            print('[DEBUG] target update')
            anchor = [_query(learningset, emb) for emb in learningset['embeddings'][:frame_idx]]
            new_target_idxes = extr.diversity(anchor=anchor)
            target = dict(names=[learningset['names'][idx] for idx in new_target_idxes], embeddings=[learningset['embeddings'][idx] for idx in new_target_idxes])
            print('[DEBUG] new targets: ' + str(new_target_idxes))
            sampler.set_target(target, weight)

    while len(sampler.models) < FirstSampler._MAX_RETRAIN:
        print("[WARN] number of etrain is less than MAX_RETRAIN.")
        #sampler.models.append(sampler.models[-1])
        result.append(result[-1])

    return result

def wrapper(args):
    return train_and_evaluate(*args)

def _query(data, qemb):
    for i, emb in enumerate(data['embeddings']):
        if (emb.reshape(128) == qemb.reshape(128)).all(): return i
    print("[WARN] data not found.")
    return -1

def add_overall_accuracy(result):
    for res in result:
        for round in res:
            round['overall'] = np.mean(round.values())

def shuffle_data(data):
    if 'pixels' in data.keys():
        names, embeddings, video, frame, pixels, orig = shuffle(data['names'], data['embeddings'], data['video'], data['frame'], data['pixels'], range(len(data['names'])))
        return dict(names=names, embeddings= embeddings, video=video, frame=frame, pixels=pixels, orig=orig)
    names, embeddings, video, frame, orig = shuffle(data['names'], data['embeddings'], data['video'], data['frame'], range(len(data['names'])))
    return dict(names=names, embeddings= embeddings, video=video, frame=frame, orig=orig)

def gen_data_frame(data, tags={}):
    columns = ['label', 'round#', 'accuracy']
    source_dict = {k: [] for k in columns}
    for d in data:
        for r_num, accuracy_dict in enumerate(d):
            for k in accuracy_dict.keys():
                source_dict['label'].append(k)
                source_dict['round#'].append(r_num)
                source_dict['accuracy'].append(accuracy_dict[k])
    for tag_name, tag_list in tags.items():
        source_dict[tag_name] = tag_list
    return pd.DataFrame.from_dict(source_dict)

#### main process ####

def eval0():
    print("Evaluation 0: Oracle")
    target_s, target_j = tutl.load_pkl(TARGET_MANUAL), tutl.load_pkl(TARGET_MANUAL_JAMES)
    d0_s, d0_j = tutl.load_pkl(DATA0_SANDEEP), tutl.load_pkl(DATA0_JAMES)
    ldata_s, ldata_j = tutl.load_pkl(LEARNING_DATA_SANDEEP), tutl.load_pkl(LEARNING_DATA_JAMES)
    oracle_s, oracle_j = tutl.load_pkl(ORACLE_SANDEEP), tutl.load_pkl(ORACLE_JAMES)

    db = tutl.load_pkl(os.path.join(DATA_PATH, 'face_db.pkl'))
    def preprocess(img, size=(100, 100)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size)
    pixels = [preprocess(db[video][frame][0]['face_pixels']) for video, frame in zip(oracle_s['video'], oracle_s['frame'])]
    extr = TargetExtractor(pixels)
    oracle_s_wp = copy.copy(oracle_s)
    oracle_s_wp['pixels'] = pixels

    num = 20

    ssim_idxes = [extr.diversity(min_tile=0, rank_by='mean') for i in range(num)]
    ssim_targets = [{'names': [oracle_s['names'][idx] for idx in ssim_idx], 'embeddings': [oracle_s['embeddings'][idx] for idx in ssim_idx], 'video': [oracle_s['video'][idx] for idx in ssim_idx], 'frame': [oracle_s['frame'][idx] for idx in ssim_idx]} for ssim_idx in ssim_idxes]

    arg1 = [(target, d0_s, shuffle_data(oracle_s), 10, 0.2, 'RAND', None, None) for target in ssim_targets]
    arg2 = [(target, d0_s, shuffle_data(oracle_s), 10, 0.2, 'DIST', None, None) for target in ssim_targets]
    arg3 = [(target, d0_s, shuffle_data(oracle_s_wp), 10, 0.2, 'DIST', None, extr) for target in ssim_targets]

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, arg1+arg2+arg3)
    pool.close()
    add_overall_accuracy(result)

    tags = dict(sampler=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10) for tag in ['random', 'distance(static)', 'distance(dynamic)']]))
    plot_data = gen_data_frame(result, tags).query('label == "sandeep"')

    #print(plot_data)
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval0_plot.pkl'), plot_data)
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval0_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='sampler', ylim=ylim, xticks=xticks)

def eval1():
    print("Evaluation 1: Accuracy vs Amount of data")
    eval_cases = ['sandeep', 'james', 'overall']
    dummy_target = tutl.load_pkl(TARGET_RSS)

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, [(dummy_target, empty, ldata, 10, 0.2, 'RAND', None, None)])
    pool.close()

    add_overall_accuracy(result)

    plot_data = {case: [round[case] for round in result[0]] for case in eval_cases}
    #print(plot_data)
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval1_plot.pkl'), plot_data)

    ts_dict = {leg: dict(ts_vector=plot_data[leg]) for leg in eval_cases}
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval1_plot.pdf')
    overlaid_ts(normalized_ts_dict =ts_dict, plot_file = plot_file, ylabel =ylabel, xlabel =xlabel, fontsize = 18, xticks = xticks, ylim = ylim, DEFAULT_ALPHA = 1.0, legend_present = True, DEFAULT_MARKERSIZE = 15, delete_yticks = False, xlim = None)

def eval2():
    print("Evaluation 2: Accuracy distribution for all labels")
    dummy_target = tutl.load_pkl(TARGET_RSS)

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, [(dummy_target, empty, shuffle_data(ldata), 10, 0.2, 'RAND', None, None) for i in range(100)])
    pool.close()

    add_overall_accuracy(result)

    plot_data = gen_data_frame(result)
    #plot_data = plot_data.query('label in ["sandeep", "james", "overall"]')
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval2_plot.pkl'), plot_data)
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval2_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='label', ylim=ylim, xticks=xticks)

def eval3():
    print("Evaluation 3: comparison of intelligent, random and oracle sampler")
    target_s, target_j = tutl.load_pkl(TARGET_MANUAL), tutl.load_pkl(TARGET_MANUAL_JAMES)
    d0_s, d0_j = tutl.load_pkl(DATA0_SANDEEP), tutl.load_pkl(DATA0_JAMES)
    #learning data
    ldata_s, ldata_j = tutl.load_pkl(LEARNING_DATA_SANDEEP), tutl.load_pkl(LEARNING_DATA_JAMES)
    #learning data for oracle
    oracle_s, oracle_j = tutl.load_pkl(ORACLE_SANDEEP), tutl.load_pkl(ORACLE_JAMES)

    num = 20
    arg1 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.2, 'RAND', None, None) for i in range(num)]
    arg2 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.2, 'DIST', None, None) for i in range(num)]
    arg3 = [(target_s, d0_s, shuffle_data(oracle_s), 10, 0.2, 'RAND', None, None) for i in range(num)]
    arg4 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.2, 'RAND', None, None) for i in range(num)]
    arg5 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.2, 'DIST', None, None) for i in range(num)]
    arg6 = [(target_j, d0_j, shuffle_data(oracle_j), 10, 0.2, 'RAND', None, None) for i in range(num)]

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, arg1+arg2+arg3+arg4+arg5+arg6)
    pool.close()

    add_overall_accuracy(result)

    tags = dict(target=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10*3) for tag in ['sandeep', 'james']]), sampler=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10) for tag in ['random', 'distance', 'oracle', 'random', 'distance', 'oracle']]))
    plot_data = gen_data_frame(result, tags).query('(label == "sandeep" and target == "sandeep") or (label == "james" and target == "james")')

    #print(plot_data)
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval3_plot.pkl'), plot_data)
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval3_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='sampler', style='target', dashes=True, ylim=ylim, xticks=xticks)

def eval4():
    print("Evaluation 4: evaluate the impact of threshold")
    target_s, target_j = tutl.load_pkl(TARGET_MANUAL), tutl.load_pkl(TARGET_MANUAL_JAMES)
    d0_s, d0_j = tutl.load_pkl(DATA0_SANDEEP), tutl.load_pkl(DATA0_JAMES)
    ldata_s, ldata_j = tutl.load_pkl(LEARNING_DATA_SANDEEP), tutl.load_pkl(LEARNING_DATA_JAMES)

    num = 20
    arg1 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.15, 'DIST', None, None) for i in range(num)]
    arg2 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.20, 'DIST', None, None) for i in range(num)]
    arg3 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.30, 'DIST', None, None) for i in range(num)]
    arg4 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.50, 'DIST', None, None) for i in range(num)]
    arg5 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.15, 'DIST', None, None) for i in range(num)]
    arg6 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.20, 'DIST', None, None) for i in range(num)]
    arg7 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.30, 'DIST', None, None) for i in range(num)]
    arg8 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.50, 'DIST', None, None) for i in range(num)]

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, arg1+arg2+arg3+arg4+arg5+arg6+arg7+arg8)
    pool.close()

    add_overall_accuracy(result)

    tags = dict(threshold=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10) for tag in ['t=0.15', 't=0.2', 't=0.3', 't=0.5', 't=0.15', 't=0.2', 't=0.3', 't=0.5']]), target=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10*4) for tag in ['sandeep', 'james']]))
    plot_data = gen_data_frame(result, tags).query('(label == "sandeep" and target == "sandeep") or (label == "james" and target == "james")')
    print(tags)

    #print(plot_data)
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval4_plot.pkl'), plot_data)
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval4_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='threshold', style='target', dashes=True, ylim=ylim, xticks=xticks)

def eval5():
    print("Evaluation 5: Learning two targets with priority")

    target_s, target_j = tutl.load_pkl(TARGET_MANUAL), tutl.load_pkl(TARGET_MANUAL_JAMES)
    target = dict(names=target_s['names'] + target_j['names'], embeddings=target_s['embeddings'] + target_j['embeddings'], video=target_s['video'] + target_j['video'], frame=target_s['frame'] + target_j['frame'])
    d0 = tutl.load_pkl(DATA0_2TARGETs)
    learningset = tutl.load_pkl(LEARNING_DATA_BOTH)

    num = 20

    arg1 = [(target, d0, shuffle_data(learningset), 10, 0.2, 'WEIGHT', dict(sandeep=1, james=1), None) for i in range(num)]
    arg2 = [(target, d0, shuffle_data(learningset), 10, 0.2, 'WEIGHT', dict(sandeep=10, james=1), None) for i in range(num)]
    arg3 = [(target, d0, shuffle_data(learningset), 10, 0.2, 'WEIGHT', dict(sandeep=1, james=10), None) for i in range(num)]

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, arg1+arg2+arg3)
    pool.close()
    #print(result)

    add_overall_accuracy(result)

    tags = dict(weight=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10) for tag in ['S1:J1', 'S10:J1', 'S1:J10']]))
    plot_data = gen_data_frame(result, tags).query('label in ["sandeep", "james", "overall"]')

    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval5_plot.pkl'), plot_data)

    plot_file = os.path.join(RESULT_OUT_PATH, 'eval5_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='label', style='weight', dashes=True, ylim=ylim, xticks=xticks)

def eval6():
    print("Evaluation 6: comparison of impact of target imageset")
    target_s, target_j = tutl.load_pkl(TARGET_MANUAL), tutl.load_pkl(TARGET_MANUAL_JAMES)
    d0_s, d0_j = tutl.load_pkl(DATA0_SANDEEP), tutl.load_pkl(DATA0_JAMES)
    ldata_s, ldata_j = tutl.load_pkl(LEARNING_DATA_SANDEEP), tutl.load_pkl(LEARNING_DATA_JAMES)
    oracle_s, oracle_j = tutl.load_pkl(ORACLE_SANDEEP), tutl.load_pkl(ORACLE_JAMES)

    db = tutl.load_pkl(os.path.join(DATA_PATH, 'face_db.pkl'))
    def preprocess(img, size=(100, 100)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size)
    pixels_s = [preprocess(db[video][frame][0]['face_pixels']) for video, frame in zip(oracle_s['video'], oracle_s['frame'])]
    extr_s = TargetExtractor(pixels_s)

    pixels_j = [preprocess(db[video][frame][0]['face_pixels']) for video, frame in zip(oracle_j['video'], oracle_j['frame'])]
    extr_j = TargetExtractor(pixels_j)
    #oracle_s['pixels'] = pixels

    num = 20

    ssim_idxes_s = [extr_s.diversity(min_tile=0, rank_by='mean') for i in range(num)]
    ssim_targets_s = [{'names': [oracle_s['names'][idx] for idx in ssim_idx], 'embeddings': [oracle_s['embeddings'][idx] for idx in ssim_idx], 'video': [oracle_s['video'][idx] for idx in ssim_idx], 'frame': [oracle_s['frame'][idx] for idx in ssim_idx]} for ssim_idx in ssim_idxes_s]

    ssim_idxes_j = [extr_j.diversity(min_tile=0, rank_by='mean') for i in range(num)]
    ssim_targets_j = [{'names': [oracle_j['names'][idx] for idx in ssim_idx], 'embeddings': [oracle_j['embeddings'][idx] for idx in ssim_idx], 'video': [oracle_j['video'][idx] for idx in ssim_idx], 'frame': [oracle_j['frame'][idx] for idx in ssim_idx]} for ssim_idx in ssim_idxes_j]

    ssimq_idxes_s = [extr_s.quantile() for i in range(num)]
    ssimq_targets_s = [{'names': [oracle_s['names'][idx] for idx in ssim_idx], 'embeddings': [oracle_s['embeddings'][idx] for idx in ssim_idx], 'video': [oracle_s['video'][idx] for idx in ssim_idx], 'frame': [oracle_s['frame'][idx] for idx in ssim_idx]} for ssim_idx in ssimq_idxes_s]

    ssimq_idxes_j = [extr_j.quantile() for i in range(num)]
    ssimq_targets_j = [{'names': [oracle_j['names'][idx] for idx in ssim_idx], 'embeddings': [oracle_j['embeddings'][idx] for idx in ssim_idx], 'video': [oracle_j['video'][idx] for idx in ssim_idx], 'frame': [oracle_j['frame'][idx] for idx in ssim_idx]} for ssim_idx in ssimq_idxes_j]

    rand_targets_s = [gen_data(oracle_s, 18) for i in range(num)]
    rand_targets_j = [gen_data(oracle_j, 18) for i in range(num)]

    arg1 = [(target_s, d0_s, shuffle_data(oracle_s), 10, 0.3, 'RAND', None, None) for i in range(num)]
    arg2 = [(target_s, d0_s, shuffle_data(ldata_s), 10, 0.3, 'DIST', None, None) for i in range(num)]
    arg3 = [(target, d0_s, shuffle_data(ldata_s), 10, 0.3, 'DIST', None, None) for target in rand_targets_s]
    arg4 = [(target, d0_s, shuffle_data(ldata_s), 10, 0.3, 'DIST', None, None) for target in ssimq_targets_s]
    arg5 = [(target, d0_s, shuffle_data(ldata_s), 10, 0.3, 'DIST', None, None) for target in ssim_targets_s]
    arg6 = [(target_j, d0_j, shuffle_data(oracle_j), 10, 0.3, 'RAND', None, None) for i in range(num)]
    arg7 = [(target_j, d0_j, shuffle_data(ldata_j), 10, 0.3, 'DIST', None, None) for i in range(num)]
    arg8 = [(target, d0_j, shuffle_data(ldata_j), 10, 0.3, 'DIST', None, None) for target in rand_targets_j]
    arg9 = [(target, d0_j, shuffle_data(ldata_j), 10, 0.3, 'DIST', None, None) for target in ssimq_targets_j]
    arg10 = [(target, d0_j, shuffle_data(ldata_j), 10, 0.3, 'DIST', None, None) for target in ssim_targets_j]

    pool = multiprocessing.Pool(num_cpus)
    result = pool.map(wrapper, arg1+arg2+arg3+arg4+arg5+arg6+arg7+arg8+arg9+arg10)
    pool.close()

    add_overall_accuracy(result)

    tags = dict(target_generated_by=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10) for tag in ['oracle', 'manual', 'random', 'ssim(quantile)', 'ssim(diversity)', 'oracle', 'manual', 'random', 'ssim(quantile)', 'ssim(diversity)']]), target=np.hstack([np.repeat(tag, num*FirstSampler._MAX_RETRAIN*10*5) for tag in ['sandeep', 'james']]))
    plot_data = gen_data_frame(result, tags).query('(label == "sandeep" and target == "sandeep") or (label == "james" and target == "james")')

    #print(plot_data)
    tutl.write_pkl(os.path.join(RESULT_OUT_PATH, 'eval6_plot.pkl'), plot_data)
    plot_file = os.path.join(RESULT_OUT_PATH, 'eval6_plot.pdf')
    ts_plot_with_distribution(plot_data, plot_file, 'round#', 'accuracy', hue='target_generated_by', style='target', ylim=ylim, xticks=xticks)

if __name__ == '__main__':
    print("FaceNet evaluation starts...")
    #eval0()
    #eval1()
    #eval2()
    #eval3()
    #eval4()
    #eval5()
    eval6()
