# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
import sys,os
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import pandas

HARVESTNET_ROOT_DIR=os.environ['HARVESTNET_ROOT_DIR']
UTILS_DIR = HARVESTNET_ROOT_DIR + '/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

DEFAULT_PRERECORDED_VIDEO = '../video_capture_examples/csandeep_amine_andrew.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-cam", "--camera_mode", type=str, default='False',
	help="whether to use onboard camera")
ap.add_argument("-prerec", "--prerecorded_video", type=str, default=DEFAULT_PRERECORDED_VIDEO,
	help="pre-recorded video")
ap.add_argument("-outprefix", "--prefix_outvideo", type=str, default='detectOutVideo',
	help="out_video_prefix")
ap.add_argument("-out", "--out_write_mode", type=str, default=False,
	help="pre-recorded video")
ap.add_argument("--results_pkl_fname", type=str, default=False,
	help="pre-recorded video")
ap.add_argument("--output_face_dir", type=str, default=False,
	help="pre-recorded video")

args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


# setup for making the output video
#########################################
# have to invert the printed frames
frame_width = 450
frame_height = 600

prefix = args['prefix_outvideo']

if args['out_write_mode'] == 'True':
	out_file = 'SVM_output_' + args['prefix_outvideo'] + '.avi'
	out = cv2.VideoWriter(out_file, fourcc, 20, (frame_height, frame_width))
	out_write_mode = True
	out_csv = 'df_SVM_output_' + args['prefix_outvideo'] + '.txt'
else:
    out_write_mode = False

# initialize the video stream, then allow the camera sensor to warm up
if args['camera_mode'] == 'True':
	print("[INFO] LIVE STREAM FROM CAMERA ")

	gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

	out_file_prefix = 'nvidia'

	vs = cv2.VideoCapture(gst_str)
	time.sleep(2.0)
else:
	print("[INFO] prerecorded_str ", args['prerecorded_video'])
	vs = cv2.VideoCapture(args['prerecorded_video'])


frame_number = 0
waitkey_duration = 1

# start the FPS throughput estimator
fps = FPS().start()

FRAME_POLL = 1

# embedding distances dict
# per name, the past embedding we had
EMBEDDING_DIMENSION =  128
past_frame = None

frame_embedding_dict = {}

FACE_WRITE_MODE = True

# loop over frames from the video file stream
while vs.isOpened():
	# grab the frame from the threaded video stream
	ret, frame = vs.read()

        if ret:

            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            # frame diff is a differencing frame !!
            if frame_number > 0:
                    frame_diff = cv2.absdiff(frame, past_frame)
            else:
                    frame_diff = frame

            frame_diff_val = np.sum(frame_diff)/(h * w * 3)

            past_frame = frame.copy()

            if frame_number % FRAME_POLL == 0:

                    # construct a blob from the image
                    imageBlob = cv2.dnn.blobFromImage(
                            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                            (104.0, 177.0, 123.0), swapRB=False, crop=False)

                    # apply OpenCV's deep learning-based face detector to localize
                    # faces in the input image
                    detector.setInput(imageBlob)
                    detections = detector.forward()

                    # initialize an empty list of detections to save
                    detected_faces_list = []

                    # loop over the detections
                    for i in range(0, detections.shape[2]):
                            # extract the confidence (i.e., probability) associated with
                            # the prediction
                            confidence = detections[0, 0, i, 2]


                            # filter out weak detections
                            if confidence > args["confidence"]:

                                # init a new dict PER detection
                                specific_face_detection_dict = {}
                                specific_face_detection_dict['frame_number'] = frame_number
                                specific_face_detection_dict['detection_confidence'] = confidence
                                specific_face_detection_dict['frame_diff_val'] = frame_diff_val

                                # compute the (x, y)-coordinates of the bounding box for
                                # the face
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                specific_face_detection_dict['box'] = [startX, startY, endX, endY]


                                # extract the face ROI
                                face = frame[startY:endY, startX:endX]
                                (fH, fW) = face.shape[:2]

                                if FACE_WRITE_MODE:
                                    # show the output frame
                                    face_path = args["output_face_dir"] + '/face_frame_' + str(frame_number) + '_detection_' + str(i) + '.jpg'
                                    cv2.imwrite(face_path, face)


                                specific_face_detection_dict['face_pixels'] = face
                                specific_face_detection_dict['face_dim'] = [fH, fW]

                                # ensure the face width and height are sufficiently large
                                if fW < 20 or fH < 20:
                                        continue

                                # construct a blob for the face ROI, then pass the blob
                                # through our face embedding model to obtain the 128-d
                                # quantification of the face
                                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                                embedder.setInput(faceBlob)
                                vec = embedder.forward()

                                specific_face_detection_dict['embedding'] = vec

                                # SAVE THE DETECTIONS AND EMBEDDINGS TO PICKLE
                                detected_faces_list.append(specific_face_detection_dict)

            frame_embedding_dict[frame_number] = detected_faces_list

            # update the FPS counter
            fps.update()

            # update frame number
            frame_number += 1

            # show the output frame
            cv2.imshow("Video", frame)

            key = cv2.waitKey(waitkey_duration) & 0xFF

            if out_write_mode:
                out.write(frame)

        else:
            write_pkl(fname = args["results_pkl_fname"], input_dict = frame_embedding_dict)
            break

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		out_file.close()
		break


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
if out_write_mode:
	out.release()

cv2.destroyAllWindows()
vs.release()
