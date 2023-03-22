import cv2
import mediapipe as mp
import time
# from cv2 import Image
import numpy as np
 
 
class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
 
        self.minDetectionCon = minDetectionCon
 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
 
    def findFaces(self, img, draw=True):
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
 
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return img, bboxs
 
    def fancyDraw(self, img, bbox, color = (255,144,30), l=30, t=4, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
 
        cv2.rectangle(img, bbox, color, rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y+l), color, t)
        # Top Right  x1,y
        # cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        # cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        # cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        # cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)
        return img
    
    def extract_face(image, box, required_size = (160,160)):

        pixels = np.asarray(image)

        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)

        return np.asarray(image)
 
 
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img, bboxs = detector.findFaces(img)
        print(bboxs)
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)


        if key == 27:
            break
 
 
if __name__ == "__main__":
    main()