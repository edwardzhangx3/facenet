# where is the neural net to detect faces
DNN_ROOT_DIR=$HARVESTNET_ROOT_DIR/DNN_models/facenet

EMBEDDING_MODEL=$DNN_ROOT_DIR/openface_nn4.small2.v1.t7

# where are the videos stored?
BASE_VIDEO_DIR=$HARVESTNET_ROOT_DIR/tmp

if [ $# -gt 0 ] ; then
	PREFIX="$1"
else
# what is the video named?
#PREFIX='james_apoorva_sandeep_1224'
	PREFIX='csandeep_no_inference'
fi

PRERECORDED_VIDEO=$BASE_VIDEO_DIR/${PREFIX}'.avi'

# write output video
OUT_WRITE_MODE='False'

# use live nvidia camera or not
CAMERA_MODE='False'

DATA_DIR=$BASE_VIDEO_DIR/EMBEDDINGS

RESULTS_PKL_FNAME=$DATA_DIR/${PREFIX}_embeddings_detections.pkl

OUTPUT_FACE_DIR=$BASE_VIDEO_DIR/face_pictures/

python get_video_embeddings.py --detector $DNN_ROOT_DIR/face_detection_model \
	--embedding-model $EMBEDDING_MODEL\
	-cam $CAMERA_MODE -outprefix $PREFIX -out $OUT_WRITE_MODE --prerecorded_video $PRERECORDED_VIDEO\
    --results_pkl_fname $RESULTS_PKL_FNAME \
    --output_face_dir $OUTPUT_FACE_DIR
