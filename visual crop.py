
import cv2
import json
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='ka42g8sMJUdBaRB13Kt8wadQbQKksbdm1mle7XA3Kixu')

# Defining a function that will do the detections
def detect(gray, frame):
    FaceFileName = "pic.jpg"
    cv2.imwrite(FaceFileName, frame)
    with open('./pic.jpg', 'rb') as images_file:
        classes = visual_recognition.classify(
        images_file,
        threshold='0.6',
        classifier_ids='Crop_1273797697').get_result()
        data=json.dumps(classes, indent=2)
        print(classes["images"][0]["classifiers"][0]["classes"][0]["class"])
        #print(data)
        return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

