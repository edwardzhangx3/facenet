# is this a LARGE or SMALL facenet model?
# LARGE: nn4.v2.t7
# SMALL: openface_nn4.small2.v1.t7
DNN_PREFIX=SMALL

# where pre-trained facenet models are
DNN_MODEL_DIR=$HARVESTNET_ROOT_DIR/DNN_models/facenet

# choose the small openface model
FACENET_DNN=$DNN_MODEL_DIR/openface_nn4.small2.v1.t7

# uncomment if we want the LARGE one
#FACENET_DNN=$DNN_MODEL_DIR/nn4.v2.t7

# code to train SVM: this directory
CODE_DIR=$HARVESTNET_ROOT_DIR/facenet/process_embeddings/

# where to put results
RESULTS_DIR=$HARVESTNET_ROOT_DIR/scratch_results/

# where image data resides
DATA_DIR=$HARVESTNET_ROOT_DIR/data/facenet/

# train an EDGE AND CLOUD model, where EDGE = ROBOT
for PREFIX in EDGE

do
    echo $PREFIX

    DATASET=${DATA_DIR}/${PREFIX}_DATASET
    echo $DATASET

    BASE_PLOT_DIR=${RESULTS_DIR}/${PREFIX}_plots/
    rm -rf ${BASE_PLOT_DIR}
    mkdir -p ${BASE_PLOT_DIR}

    echo 'start analyzing embeddings'

    SVM_RESULTS_DIR=${DNN_MODEL_DIR}/${DNN_PREFIX}_${PREFIX}_SVM/

    python -i $CODE_DIR/plot_embedding_distance_per_person.py --embeddings $SVM_RESULTS_DIR/embeddings.pickle --base-plot-dir $BASE_PLOT_DIR
done
