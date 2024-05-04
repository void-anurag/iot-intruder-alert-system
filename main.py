#import the necessary packages
from dotenv import load_dotenv
load_dotenv()
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import uuid
import imutils
import time
import cv2
from time import sleep
import telebot
import datetime
import os
image_folder = "sus"
BOT_TOKEN = os.environ.get('BOT_API_KEY')
CHAT_ID=os.environ.get('CHAT_ID')

bot = telebot.TeleBot(BOT_TOKEN)
image_sent = False

# Set default values
DEFAULT_PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
DEFAULT_MODEL = "MobileNetSSD_deploy.caffemodel"
DEFAULT_CONFIDENCE = 0.2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default=DEFAULT_PROTOTXT,
                help="path to Caffe 'deploy' prototxt file (default: %(default)s)")
ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                help="path to Caffe pre-trained model (default: %(default)s)")
ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE,
                help="minimum probability to filter weak detections (default: %(default)s)")
args = vars(ap.parse_args())

# board = Arduino('COM3')#usb port
# pin = [3,5,6,9]


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            percent = confidence * 100
            label = "{}: {:.2f}%".format(CLASSES[id],percent)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id], 2)

        if CLASSES[id] == "person" and percent > 99.5 and not image_sent:
            # Set the flag to True to indicate that an image has been sent
            image_sent = True

            # Generate a unique filename for the image
            unique_id = str(uuid.uuid4())[:8]  # Generate a random 8-character UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")  # Get current timestamp
            image_filename = os.path.join(image_folder, f"image_{timestamp}_{unique_id}.jpg")

            # Create a caption for the image
            caption = "An intruder was spotted snooping around your room."

            # Save the frame as a JPEG image
            cv2.imwrite(image_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Send the image to the user
            with open(image_filename, "rb") as photo_file:
                bot.send_photo(chat_id=CHAT_ID, photo=photo_file,caption=caption)
            print("server sent the image")

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# cd\
# cd C:\Users\cocsa\OneDrive\Documents\VS\Python\robotics\Complete
# python main.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
