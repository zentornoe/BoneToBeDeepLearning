from scipy.spatial import distance
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

images = glob('deeplearingtxt/*.png')

features = []
labels = []
for im in images:
    labels.append(im[15:-len('.png')])
    im = mh.imread(im)
    #im = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(mh.features.haralick(im).ravel())

features = np.array(features)
labels = np.array(labels)

clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])

cv = model_selection.LeaveOneOut()

scores = model_selection.cross_val_score(clf, features, labels, cv=cv)
print(scores)
print('Accuracy: {:.2%}'.format(scores.mean()))

sc = StandardScaler()
features = sc.fit_transform(features)
print(features)
dists = distance.squareform(distance.pdist(features))


def selectImage(n, m, dists, images):
    image_position = dists[n].argsort()[m]
    image = mh.imread(images[image_position])
    return image


def plotImages(n):
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(selectImage(n, 0, dists, images))
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(selectImage(n, 1, dists, images))
    plt.title('1st simular one')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    plt.imshow(selectImage(n, 2, dists, images))
    plt.title('2nd simular one')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    plt.imshow(selectImage(n, 3, dists, images))
    plt.title('3rd simular one')
    plt.xticks([])
    plt.yticks([])

    plt.show()

plotImages(2)
