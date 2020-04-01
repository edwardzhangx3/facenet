# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

DEFAULT_PRERECORDED_VIDEO = '../video_capture_examples/csandeep_amine_andrew.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
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

ap.add_argument("--base_output_video_dir", type=str, default=False,
	help="pre-recorded video")

args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup

frame_width = 300
frame_height = 400

if args['out_write_mode'] == 'True':
	out_file = args['base_output_video_dir'] + '/output_' + args['prefix_outvideo'] + '.avi'
	out = cv2.VideoWriter(out_file, fourcc, 20, (frame_height, frame_width))
	out_write_mode = True


if args['camera_mode'] == 'True':
	print("[INFO] LIVE STREAM FROM CAMERA ")

	gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

	out_file_prefix = 'nvidia'

	vs = cv2.VideoCapture(gst_str)
	time.sleep(2.0)
else:
	print("[INFO] prerecorded_str ", args['prerecorded_video'])
	vs = cv2.VideoCapture(args['prerecorded_video'])


#frame_width = int( vs.get(cv2.CAP_PROP_FRAME_WIDTH))
#frame_height =int( vs.get( cv2.CAP_PROP_FRAME_HEIGHT))




waitkey_duration = 1

# loop over the frames from the video stream
#while True:

frame_number = 0

FRAME_POLL = 1

while vs.isOpened():



	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=400)
	#frame = cv2.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]


	if frame_number % FRAME_POLL == 0:

		# don't do this every frame
		##################################################
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < args["confidence"]:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	frame_number += 1

	cv2.imshow("Frame", frame)
	#cv2.imshow("Video", frame)
	if out_write_mode:
		out.write(frame)

	if cv2.waitKey(waitkey_duration) & 0xFF == ord('q'):
		break


# do a bit of cleanup
if out_write_mode:
	out.release()
cv2.destroyAllWindows()
vs.stop()
