from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets, svm, metrics
from matplotlib.pyplot import imread
from fnmatch import fnmatch as match
from zipfile import ZipFile
from util import rgb2gray
from cv2 import resize
import wget, os, re
from hog import hog
import numpy as np
import pickle
import sys

img_w = 512
img_h = 512

dictionary = {
    "f" : "fear",
    "a" : "angry",
    "d" : "disgusted",
    "h" : "happy",
    "n" : "neutral",
    "sa" : "sad",
    "su" : "surprised"
}

print("Training facial emotion model")

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python train_emotion.py [database_name] [traing split percent]")
        print("Example: python3.8 train_emotion database 0.7")
        print("The database and this file should be in the same directory")
        sys.exit()

    database_name = sys.argv[1]
    train_precent = sys.argv[2]
    data_path = './' + database_name

    hog = hog()
    hog.summary()
    
    print("Processing dataset")
    labels = []
    data = np.zeros((1, int(hog.feature_size * 64))) # Create empty image so the other can stack here
    for emotion in os.listdir(data_path):
        for image in os.listdir(data_path + '/' + emotion):
            img = imread(data_path + '/' + emotion + '/' + image)
            img = resize(img, (img_w, img_h))
            img = rgb2gray(img)
            feature = []
            for i in range(0, img_w, 64):
                for j in range(0, img_h, 64):
                    patch = img[i:i+64, j:j+64]
                    feature = np.concatenate((feature, [hog.get_feature(patch)]), axis=None)
            data = np.vstack((data,feature))
            feature = [hog.get_feature(img)]
            labels.append(list(dictionary).index(emotion))
    data = data[1:] # Delete the dummy feature
    labels = np.array(labels)

    classifier = svm.SVC()
    data_train, data_test, labels_train, label_test = train_test_split(
        data, labels, test_size=float(train_precent), shuffle=False)

    print("Training the model")
    classifier.fit(data_train, labels_train)
    pickle.dump(classifier, open((database_name + '.svm'), 'wb'))

    print("Testing the model")
    predicted = classifier.predict(data_test)
    
    f = open(database_name + "_report.txt", "a")
    f.write("Classification report for classifier\n")
    f.write(str(metrics.classification_report(label_test, predicted)))
    f.write(str(confusion_matrix(label_test, predicted)))
    f.close()
