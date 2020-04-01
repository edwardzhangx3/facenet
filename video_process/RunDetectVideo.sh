# where is the neural net to detect faces
DNN_ROOT_DIR=$HARVESTNET_ROOT_DIR/DNN_models/facenet/face_detection_model

# where are the videos stored?
BASE_VIDEO_DIR=/Users/edward/Desktop/

# what is the video named?
PREFIX='james_apoorva_sandeep_1224'

PRERECORDED_VIDEO=$BASE_VIDEO_DIR/${PREFIX}'.avi'

BASE_OUTPUT_VIDEO_DIR=$BASE_VIDEO_DIR/WITH_DETECTIONS

# write output video
OUT_WRITE_MODE='True'

# use live nvidia camera or not
CAMERA_MODE='False'

python detect_faces_video.py --prototxt $DNN_ROOT_DIR/deploy.prototxt \
	--model $DNN_ROOT_DIR/res10_300x300_ssd_iter_140000.caffemodel -cam $CAMERA_MODE -outprefix $PREFIX -out $OUT_WRITE_MODE --prerecorded_video $PRERECORDED_VIDEO --base_output_video_dir $BASE_OUTPUT_VIDEO_DIR
