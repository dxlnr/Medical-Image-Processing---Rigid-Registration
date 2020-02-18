import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib import pyplot as plt
import utils

from scipy import ndimage as ndi
from skimage import transform as tf
from skimage import feature
from skimage import color
from skimage import filters
from skimage import exposure
import cv2


'''
    Part (A) – Point and image based registration algorithms.
'''

bl = plt.imread('data/BL01.tif')
fu = plt.imread('data/FU01.tif')

BLPoints, FUPoints = utils.getPoints('no_outliers')

def pickCO(dataarray, row):
    output = []
    if (row == 'x'):
        for i in range(BLPoints.shape[0]):
            tmp = int(dataarray[i][0])
            output.append(tmp)
    elif (row == 'y'):
        for i in range(BLPoints.shape[0]):
            tmp = int(dataarray[i][1])
            output.append(tmp)
    elif (row == 'z'):
        for i in range(BLPoints.shape[0]):
            tmp = int(dataarray[i][2])
            output.append(tmp)
    else:
        print("Unknown string " + row)
        exit()

    return (output)


def plotting_points(image, pointset, number=True):
    '''
    Plotting the points with their labeling in the image.
    :param image: Input image.
    :param pointset: Pointset you want to plot. -- shape (N,2)
    :param number: Set to False if you do not want to label the points.
    '''
    plt.imshow(image)
    for idx in range(pointset.shape[0]):
        x = pointset[idx][0]
        y = pointset[idx][1]
        plt.plot(x, y, 'bo')
        if number:
            plt.text(x + 2, y + 2 , idx, fontsize=13)


def plotting_pairs(BLPoints, FUPoints):
    '''
    Plotting the point pairs next to each other in one plot.
    :param BLPoints: Baseline points -- shape (N,2)
    :param FUPoints: Follow-up points -- shape (N,2)
    Function for plotting pairs of points in two transformed images.
    '''
    fig = plt.figure()
    a1 = fig.add_subplot(1, 2, 1)
    plotting_points(bl, BLPoints)

    a2 = fig.add_subplot(1, 2, 2)
    plotting_points(fu, FUPoints)

    for idx in range(BLPoints.shape[0]):
        xbl = BLPoints[idx][0]
        ybl = BLPoints[idx][1]
        xfu = FUPoints[idx][0]
        yfu = FUPoints[idx][1]

        con = ConnectionPatch(xyA=(xfu,yfu), xyB=(xbl,ybl), coordsA="data", coordsB="data", axesA=a2, axesB=a1, color="red")
        a2.add_artist(con)

    plt.show()

plotting_pairs(BLPoints, FUPoints)


class PointbasedReg():

    def __init__(self, robust=True, pointstring='no_outliers'):
        '''
        Loads two images – BL and FL – and a set of chosen point, computes the rigid registration transformation between them
        and applies it to the FU image.
        '''
        self.bl = plt.imread('data/BL01.tif')
        self.fu = plt.imread('data/FU01.tif')
        self.BLPoints, self.FUPoints = utils.getPoints(pointstring)

        if robust == True:
            self.tform, self.inliers = self.calcRobustPointBasedReg(self.BLPoints, self.FUPoints)
        else:
            self.tform = self.calcPointBasedReg(self.BLPoints, self.FUPoints)

        self.affineTransformation(self.bl, self.fu, self.tform)
        print(self.tform)
        #rmse = self.calcDist(self.BLPoints, self.FUPoints, self.tform)
        #print("rmse of rigid registration: ", rmse)


    def calcPointBasedReg(self, BLPoints, FUPoints):
        '''
        :param BLPoints: Baseline points -- shape (N,2)
        :param FUPoints: Follow-up points -- shape (N,2)
        :return: A 3x3 rigid 2D transformation matrix of the two translations and rotations of the given points and pairings
        '''
        assert len(BLPoints) == len(FUPoints)
        # Preprocess
        result = np.zeros((3,3), dtype=float)
        np.fill_diagonal(result, 1.0)

        # 1. Compute the (weighted) centroids of both point sets
        meanBL = np.mean(BLPoints, axis=0)
        meanFU = np.mean(FUPoints, axis=0)

        # 2. Compute the centered vectors
        centeredBL = BLPoints - np.tile(meanBL, (BLPoints.shape[0], 1))
        centeredFU = FUPoints - np.tile(meanFU, (FUPoints.shape[0], 1))

        # 3. Compute the dxd covariance matrix
        cov = np.matmul(np.transpose(centeredBL), centeredFU)

        # 4. Compute the singular value decomposition
        U, S, Vt = np.linalg.svd(cov)
        R = np.matmul(Vt.T, U.T)

        # 5. Compute the optimal translation
        t = np.add(np.dot(np.matmul(R, np.transpose(meanBL)), -1), np.transpose(meanFU))

        # Create result rigid matrix
        result[0:R.shape[0], 0:R.shape[1]] = R
        result[0:t.shape[0], R.shape[0]] = t.T

        return (result)

    def calcDist(self, BLPoints, FUPoints, rigidReg):
        '''
        The resulting matrix should approximately satisfy the following relation: [BLPoints ones(N,1)] * tform == [FUPoints ones(N,1)]
        :param BLPoints: Baseline points -- shape (N,2)
        :param FUPoints: Follow-up points -- shape (N,2)
        :param rigidReg: transfomation matrix. (3,3)
        :return: scalar rmse vector -- shape (N,)
        '''
        # Create 3x3 matrix
        helper = np.ones((BLPoints.shape[0],1), dtype=float)
        testBL = np.concatenate((BLPoints, helper), axis=1)

        rigidBL = np.matmul(rigidReg, testBL.T)
        rigidBL = np.delete(rigidBL.T, 2, axis=1)

        rmse = np.sqrt(np.mean(np.square(rigidBL - FUPoints), axis=1))

        return (rmse)

    def affineTransformation(self, blimage, fuimage, rigidReg, plot=True):
        '''
        The function will compute a new image of the transformed FU image.
        Set option for plotting transformed FU image overlaided by the BL image.
        :param plot: default=True.
        '''
        transformed = tf.warp(fuimage, rigidReg)

        if plot == True:
            plt.figure()
            plt.imshow(blimage)
            plt.imshow(transformed, cmap='Greens', alpha=0.5)
            plt.show()


    def calcRobustPointBasedReg(self, BLPoints, FUPoints):
        '''
        This function computes the transformation with unknown outliers in the pairs list.
        '''
        # Setup the parameters for ransac
        minPtNum = 3
        iterNum = 10
        thDist = 20
        thInlrRatio = 0.5

        tform, inlieridx = utils.ransac(BLPoints, FUPoints, self.calcPointBasedReg, self.calcDist,
        minPtNum, iterNum, thDist, thInlrRatio)

        return (tform, inlieridx)

ptBasedReg = PointbasedReg(robust=False, pointstring='no_outliers')

'''
    Part (B) – Point and image based registration algorithms.
'''

def segmentBloodVessel(image):
    '''
    Performs segmentation of the blood vessels in the retina image.
    :param image: Input image. -- shape (1636, 1536, 3)
    :return: Segmented binary image. -- shape (1636, 1536)
    '''
    grey_image = color.rgb2gray(image)

    # Apply median filter in order to decrease noise.
    median = filters.median(grey_image)

    # Apply CLAHE to enhance contrast.
    clahe = exposure.equalize_adapthist(median)

    # Apply the frangi filter on the preprocessed image in order to find vessel structures.
    out = filters.frangi(clahe, sigmas=range(4,9,1))

    # Thresholding with the frangi filtered image to create binary.
    mask = np.where(out >= 1e-5)
    out[mask] = 1

    # Cut of the black caption on the bottom of the image.
    roi = np.zeros((grey_image.shape), dtype=bool)
    roi[50:1525,50:1510] = True
    out[~roi] = 0

    # Apply an morphological opening to decrease noise.
    erosion = cv2.erode(out, np.ones((3, 3), np.uint8), iterations=1)
    result = cv2.dilate(erosion, np.ones((3, 3), np.uint8), iterations=1)

    return (result)


def findRetinaFeatures(image, n_kp, method='orb', plot=True):
    '''
    The function finds strong features in the image used for registration.
    :param image: input image. -- (1636, 1536, 3)
    :param n_kp: Number of keypoints you want to find.
    :param method: Set which algorithm should be used to find keypoints.
    :param plot: Set True, found keypoints will be plotted.
    :return keypoints and descriptors representing the found features.
    '''
    # Creating a gray image for the ORB method and creating a ROI in order to ignore the black caption at the bottom.
    grey_image = color.rgb2gray(image)
    roi = grey_image[0:1300,0:1300]
    roi = filters.gaussian(roi)

    # Applying the ORB method and extracting the features.
    orb = feature.ORB(n_keypoints=n_kp)
    orb.detect_and_extract(roi)
    keypoints = orb.keypoints
    descriptors = orb.descriptors

    # Plotting option.
    if plot == True:
        plotting_points(image, keypoints, number=False)
        plt.show()

    return (keypoints, descriptors)

#keypoints, descriptors = findRetinaFeatures(bl, 50)
#segmented_vessels = segmentBloodVessel(bl)
