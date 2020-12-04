 from imutils.video import WebcamVideoStream as cam
 from util import rgb2gray

def adjust_gamma(img, gamma):
        invGamma = 1.0/ gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8"    )

        return cv2.LUT(img, table)

labels = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

face_classifier = pickle.load(open('face.svm', 'rb'))
emotion_classifier = pickle.load(open('emotion.svm', 'rb'))

hog = hog()
hog.summary()

vs =cam(src=0).start()
cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
while(True):
        frame = vs.read()
        frame = cv2.resize(frame,(512,512))
        frame = adjust_gamma(frame, 2)
        frame = rgb2gray(frame)
        feature = np.array([hog.get_feature(frame)])
        result = emotion_classifier.predict(feature)
        print(result)
        cv2.imshow("Stream",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()
vs.stop()