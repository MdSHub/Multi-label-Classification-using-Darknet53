import numpy
import argparse
import cv2
import os  


gu = argparse.ArgumentParser()
gu.add_argument("-i", "--image", required=True,
	help="input image")
gu.add_argument("-y", "--configuration", required=True,
	help="directory")
gu.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability for detections")
gu.add_argument("-t", "--threshold", type=float, default=0.3,
	help="non-maxima suppression threshold")
args = vars(gu.parse_args())


labelsPath = os.path.sep.join([args["configuration"], "coco_dataset.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
numpy.random.seed(42)
COLORS = numpy.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = os.path.sep.join([args["configuration"], "network.weights"])
configPath = os.path.sep.join([args["configuration"], "network.cfg"])


print("starting Model Darknet53")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image = cv2.imread(args["image"])
(H, W) = image.shape[:2]


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)

layerOutputs = net.forward(ln)






boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = numpy.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * numpy.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)


#on command prompt
# python label.py --image images/class.jpg --label configuration
