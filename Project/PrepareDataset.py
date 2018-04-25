import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from skimage import measure
import glob
from sklearn.decomposition import PCA

def GetDataset(TrainingDatasetPath , TestingDatasetPath):
    tmp_x_train = np.full((25,2500),0)
    y_train = np.full((25,5),0)
    classes = []
    idx=0
    for filename in glob.glob(TrainingDatasetPath + '/*.jpg'): 
        img = cv2.imread(filename,0)
        GrayImage = cv2.resize(img, (50, 50)) 
        tmp_x_train[idx,:] = np.array(GrayImage).reshape((1,2500))
        image = filename[len(TrainingDatasetPath)+2:]
        if image.split("- ")[1][:-4] not in classes:
            classes.append(image.split("- ")[1][:-4])
        y_train[idx,classes.index(image.split("- ")[1][:-4])] = 1
        idx = idx + 1
    clf=PCA(0.99,whiten=True)     #converse 90% variance
    X_train=clf.fit_transform(tmp_x_train)
       
    tmp_x_test = np.full((26,2500),0)
    y_test = np.full((26,5),0)
    idx=0
    for filename in glob.glob(TestingDatasetPath + '/*.jpg'): 
        img = cv2.imread(filename,0)
        GrayImage = cv2.resize(img, (50, 50)) 
        tmp_x_test[idx,:] = np.array(GrayImage).reshape((1,2500))
        image = filename[len(TrainingDatasetPath)+2:]
        y_test[idx,classes.index(image.split("- ")[1][:-4])] = 1
        idx = idx + 1        
    X_test=clf.transform(tmp_x_test)
    return X_train , y_train , X_test , y_test
'''       
for image in TestingImages:
    RealImage = Image.open(DatasetPath + "Testing\\" + image + ".jpg")
    plt.figure()
    plt.imshow(RealImage)
    plt.show()
    RealImage = RealImage.convert('L')
    ColoredImage = Image.open(DatasetPath + "Testing\\" + image + ".png").convert('L')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cannyimg = cv2.dilate(cv2.Canny(np.array(ColoredImage),20,20) , kernel, iterations = 1)
    SigmentedImage = cv2.bitwise_not(cannyimg)
    labels = measure.label(SigmentedImage)
    regions = measure.regionprops(labels)
    for region in regions:
        r0, c0, r1, c1 = region.bbox
        if not( ((r1-r0)*(c1-c0))/(len(np.array(RealImage)[:,0]) * len(np.array(RealImage)[0,:]))>3/4 ):
            plt.figure()
            plt.imshow(np.array(RealImage)[r0:r1,c0:c1])
            plt.show()
            cv2.imshow('image',GrayImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
'''