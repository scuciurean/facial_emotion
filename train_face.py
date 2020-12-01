from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.image import PatchExtractor
from skimage import color, data, transform, img_as_ubyte
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from cv2 import resize
from hog import hog
import numpy as np
import pickle
import os

def extract_patches(img, N, scale=1.0, patch_size=(64,64)):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                for patch in patches])
    return patches

# Window dimensions
w = 64
h = 64

# Use 80% of data to train and the rest to test
train_data_percent = 80 / 100

hog = hog()
hog.summary()

print("Training facial detection model")

print("Loading facial image dataset")
faces = fetch_lfw_people()
faces = faces.images

print("Computing positive features")
positive_features = np.vstack([hog.get_feature(faces[i]) for i in range(0, int(faces.shape[0] * train_data_percent))])
print(positive_features.shape)

print("Loading random image dataset")
labels = ['camera', 'text', 'coins', 'moon', 'page', 'clock', 'coffee']
random_img = [color.rgb2gray(getattr(data, name)()) for name in labels]
not_faces = np.vstack([extract_patches(im, 1000, scale, (w,h)) for im in random_img for scale in [0.5, 1.0, 2.0]])

print("Computing negative features")
negative_features = np.vstack([hog.get_feature(not_faces[i]) for i in range(0, int(not_faces.shape[0] * train_data_percent))])

# Prepare the dataset
data = np.concatenate((positive_features, negative_features))
labels = np.zeros(data.shape[0])
labels[:positive_features.shape[0]] = 1

classifier = svm.SVC()
print("Training the model")
classifier.fit(data, labels)
pickle.dump(classifier, open('face.svm', 'wb'))

print("Testing the model")

print("Computing positive features")
positive_features = np.vstack([hog.get_feature(faces[i]) for i in range(int(faces.shape[0] * train_data_percent), faces.shape[0])])

print("Computing negative features")
negative_features = np.vstack([hog.get_feature(not_faces[i]) for i in range(int(not_faces.shape[0] * train_data_percent), not_faces.shape[0])])

# Prepare the dataset
data = np.concatenate((positive_features, negative_features))
labels = np.zeros(data.shape[0])
labels[:positive_features.shape[0]] = 1

predicted = np.zeros(labels.shape[0])

for i in range(0, data.shape[0]):
    feature = np.array([hog.get_feature(data[i])])
    result = classifier.predict(feature)
    if (result[0] != 0):
        predicted[i] = 1

print(confusion_matrix(labels, predicted))z
print("Done")