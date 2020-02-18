import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import utils

from scipy import ndimage as ndim
from scipy.spatial.distance import cdist
from scipy import signal
from sklearn.preprocessing import normalize

from skimage import transform as tf
from skimage import feature
from skimage import filters
from skimage import color
from skimage import exposure
from skimage import segmentation
from skimage import morphology

import cv2
import imutils


'''
    Part 1 – Registration by features.
'''

class FeatureRegistrationModel():

    def __init__(self, algorithm='first'):
        self.bl = plt.imread('data/BL01.tif')
        self.fu = plt.imread('data/FU01.tif')

        if algorithm == 'first':
            self.firstalgorithm(self.bl, self.fu)
        else:
            self.secondalgorithm()



    def firstalgorithm(self, blimage, fuimage):
        '''
        Algorithm for computing registration by features using FindRetinalFeatures.
        :param blimage: Baseline image. -- shape (1636, 1536, 3)
        :param fuimage: Follow-up image. -- shape (1636, 1536, 3)
        '''
        method = 'orb'
        number_kp = 500
        bl_kp, bl_descriptors = self.findRetinaFeatures(blimage, number_kp, method=method, plot=False)
        fu_kp, fu_descriptors = self.findRetinaFeatures(fuimage, number_kp, method=method, plot=False)

        # Find matches for each specific method and keep only the 10 with the lowest distance measure.
        best_matches = 6
        if method == 'orb':
            matches, idx_list = self.match_descriptors(bl_descriptors, fu_descriptors, best_matches, metric='hamming')
        elif method == 'surf':
            matches, idx_list = self.match_descriptors(bl_descriptors, fu_descriptors, best_matches, metric='euclidean')
        elif method == 'sift':
            matches, idx_list = self.match_descriptors(bl_descriptors, fu_descriptors, best_matches, metric='euclidean')
        else:
            print("Method not found.")

        # Preparing the keypoints to feed into RANSAC.
        prepbl = []
        prepfu = []
        for idx in range(best_matches):
            tmp = matches[idx_list[idx]]
            prepbl.append(bl_kp[tmp[0]])
            prepfu.append(fu_kp[tmp[1]])

        prepbl = np.array(prepbl)
        prepfu = np.array(prepfu)

        prepbl = np.flip(prepbl, axis=1)
        prepfu = np.flip(prepfu, axis=1)

        self.plotting_pairs(prepbl, prepfu)

        tformR, inliers = self.calcRobustPointBasedReg(prepbl, prepfu)
        print("Transformation matrix: ", tformR)
        rmse = self.calcDist(prepbl, prepfu, tformR)
        print("rmse of rigid registration: ", rmse)
        print("inliers: ", inliers)

        # Transfrom and show the results.
        self.affineTransformation(blimage, fuimage, tformR, plot=True)
        plt.show()

    def match_descriptors(self, bl_descriptors, fu_descriptors, k, metric=None):
        '''
        Brute-force matching of descriptors. Works for opencv as well as skimage.
        :param bl_descriptors:
        :param fu_descriptors:
        :param k:
        :param metric:
        :return: matches and a sorted list of indices of the matches which have the lowest distance.

        '''
        if metric is None:
            if np.issubdtype(bl_descriptors.dtype, np.bool_):
                metric = 'hamming'
            else:
                metric = 'euclidean'

        distances = cdist(bl_descriptors, fu_descriptors, metric='hamming')

        indices1 = np.arange(bl_descriptors.shape[0])
        indices2 = np.argmin(distances, axis=1)

        smallest_dst = np.amin(distances, axis=1)
        idx_list = np.argpartition(smallest_dst, k)

        matches = np.column_stack((indices1, indices2))

        return (matches, idx_list)


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

    def calcRobustPointBasedReg(self, BLPoints, FUPoints):
        '''
        This function computes the transformation with unknown outliers in the pairs list.
        '''
        # Setup the parameters for ransac
        minPtNum = 3
        iterNum = 100
        thDist = 25
        thInlrRatio = 0.5

        tform, inlieridx = utils.ransac(BLPoints, FUPoints, self.calcPointBasedReg, self.calcDist,
        minPtNum, iterNum, thDist, thInlrRatio)

        return (tform, inlieridx)


    def findRetinaFeatures(self, image, n_kp, method='orb', plot=True):
        '''
        The function finds strong features in the image used for registration.
        :param image: input image. -- (1636, 1536, 3)
        :param n_kp: Number of keypoints you want to find.
        :param method: Set which algorithm should be used to find keypoints.
        :param plot: Set True, found keypoints will be plotted.
        :return keypoints and descriptors representing the found features.
        '''
        # Creating a gray image for the ORB method and creating a ROI in order to ignore the black caption at the bottom.
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = grey_image[100:1300,100:1300]

        # Applying the ORB method and extracting the features.
        if method == 'orb':
            # Preprocessing for ORB
            roi = filters.gaussian(roi)

            orb = feature.ORB(n_keypoints=n_kp)
            orb.detect_and_extract(roi)

            keypoints = orb.keypoints
            descriptors = orb.descriptors

        # Applying the SIFT method and extracting the features.
        elif method == 'sift':
            roi = cv2.GaussianBlur(roi, (5,5), cv2.BORDER_DEFAULT)
            sift = cv2.xfeatures2d.SIFT_create(n_kp)
            kp_cv, descriptors = sift.detectAndCompute(roi, None)

            # Convert opencv data structure to numpy array.
            keypoints = np.array([k.pt for k in kp_cv])

        # Applying the SURF method and extracting the features.
        elif method == 'surf':
            roi = cv2.GaussianBlur(roi, (5,5), cv2.BORDER_DEFAULT)
            surf = cv2.xfeatures2d.SURF_create(n_kp)
            kp_cv, descriptors = surf.detectAndCompute(roi, None)

            # Convert opencv data structure to numpy array.
            keypoints = np.array([k.pt for k in kp_cv])

        else:
            print("Unknown string " + method)
            exit()

        # Plotting option.
        if plot == True:
            self.plotting_points(image, keypoints, number=False)

        return (keypoints, descriptors)

    def plotting_points(self, image, pointset, number=True):
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

    def plotting_pairs(self, BLPoints, FUPoints):
        '''
        Plotting the point pairs next to each other in one plot.
        :param BLPoints: Baseline points -- shape (N,2)
        :param FUPoints: Follow-up points -- shape (N,2)
        Function for plotting pairs of points in two transformed images.
        '''
        fig = plt.figure(1)
        a1 = fig.add_subplot(1, 2, 1)
        self.plotting_points(self.bl, BLPoints, number=True)

        a2 = fig.add_subplot(1, 2, 2)
        self.plotting_points(self.fu, FUPoints, number=True)

        for idx in range(BLPoints.shape[0]):
            xbl = BLPoints[idx][0]
            ybl = BLPoints[idx][1]
            xfu = FUPoints[idx][0]
            yfu = FUPoints[idx][1]

            con = ConnectionPatch(xyA=(xfu,yfu), xyB=(xbl,ybl), coordsA="data", coordsB="data", axesA=a2, axesB=a1, color="red")
            a2.add_artist(con)


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


    def secondalgorithm(self):
        '''
        Algorithm for computing registration by features using segmentBloodVessel.
        '''
        # Perform Segmentation.
        binaryBL = self.segmentBloodVessel(self.bl)
        binaryFU = self.segmentBloodVessel(self.fu)

        # Preprocess the images for template matching.
        grey_bl = color.rgb2gray(self.bl)
        roibl = grey_bl[50:1300,50:1300]
        roibl = filters.gaussian(roibl)

        grey_fu = color.rgb2gray(self.fu)
        roifu = grey_fu
        roifu = filters.gaussian(roifu)

        # Loop to find the rotation.
        found = None
        for angle in np.linspace(-10.0, 10.0, 20)[::-1]:
            print(angle)

            rotated = tf.rotate(roibl, angle)
            result = feature.match_template(roifu, rotated)

            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, angle)
                print(found)

        (_, maxLoc, angle) = found


        rotated_seg = tf.rotate(binaryFU, angle=angle)

        #shifts, error, phasediff = feature.register_translation(self.bl, test)
        shifts, error, phasediff = feature.register_translation(rotated_seg, binaryBL)

        shifts = np.flip(shifts)
        tform = np.zeros((3,3), dtype=float)
        np.fill_diagonal(tform, 1.0)
        tform[0:shifts.shape[0], 2] = shifts.T

        transformed = tf.warp(self.fu, tform)


        plt.figure(1)
        plt.imshow(self.bl)
        plt.imshow(transformed, cmap='Greens', alpha=0.5)
        plt.show()


    def segmentBloodVessel(self, image):
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


#model = FeatureRegistrationModel('first')


'''
    Part 2 – Detect the changes.
'''


class ChangeRegistrationModel():

    def __init__(self):
        self.bl = plt.imread('data/BL01.tif')
        self.fu = plt.imread('data/FU01.tif')
        self.rgfu = plt.imread('data/rg-FU-01.png')


    def detectingChange(self):
        '''
        The function detects the changes of the lesions between two registered ophthalmology 2D images.
        '''
        rgfu = self.rgfu[:, :, 0]
        bl = self.bl[:, :, 0]
        result = np.zeros(bl.shape)

        roibl = bl[:1300,:]
        roirgfu = rgfu[:1300,:]

        mask = self.segmentLesion()

        # 0. Preprocess by apply Gaussian on the images.
        nbl = filters.gaussian(roibl)
        nrgfu = filters.gaussian(roirgfu)

        # 1. Normalize the bl image by columns.
        nbl = normalize(nbl, axis=0, norm='max')

        # 2. Subtract FU from BL after registration
        subtracted = np.subtract(nbl, nrgfu)

        # 3. Remove all pixels that are under 0 and take the absolute value of the results
        threshold_value = 0.6
        subtracted[subtracted < 0] = 0
        subtracted[subtracted > threshold_value] = 0
        subtracted = np.abs(subtracted)

        # Applying thresholding method on the image (Yen in this case.)
        yen = filters.threshold_yen(subtracted)
        subtracted[subtracted <= yen] = 0
        subtracted[subtracted >= yen] = 1

        # Clear the boarders
        cleared = segmentation.clear_border(subtracted)

        # Do some morphological operations (CLosing)
        selem = morphology.disk(4)
        closing = morphology.binary_closing(cleared, selem)


        # Apply the precomputed mask.
        result[np.where(closing == 1)] = 1
        result[~mask] = 0

        plt.figure()
        plt.imshow(self.bl)
        plt.imshow(result, cmap='Greens', alpha=0.5)
        plt.show()


    def segmentLesion(self, plotting=False):
        '''
        Helper function for finding the ROI to improve the detectingChange algorithm.
        Works with the Bottom-Line image.
        :return: circle mask which defines the ROI.
        '''
        # 0. Preprocessing or setting up the images.
        bl = self.bl[:, :, 0]
        roibl = bl[50:1300,50:1300]
        result = bl.copy()

        # 0. Preprocess by apply Gaussian on the images.
        roibl = filters.gaussian(roibl)

        # Segmentation via Thresholding (Yen-method)
        threshold_bl = filters.threshold_yen(roibl)
        roibl[roibl <= threshold_bl] = 1
        roibl[roibl != 1] = 0

        # Clear the boarders.
        cleared_bl = segmentation.clear_border(roibl)

        # Apply a morphological closing.
        selem = morphology.disk(4)
        closing_bl = morphology.binary_closing(cleared_bl, selem)

        # Find the left contours.
        contourimg = np.uint8(closing_bl)
        _, contours, _ = cv2.findContours(contourimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the biggest contour to the top of the list.
        contours.sort(key=lambda x:cv2.contourArea(x), reverse=True)

        contour_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area == 0:
                print("No area found.")
                break
            # Find the centerpoint of the segmented contour (lesion)
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = [cX, cY]

            # Compute an extrempoint in order to compute the radius of the ROI.
            extrempoint = list(contour[contour[:, :, 0].argmax()][0])
            dist = np.linalg.norm(np.array(extrempoint) - np.array(center))

            # Just keep the biggest area (which should be the lesion).
            break

        # create roi mask.
        r = dist + 50
        roi = self.create_circular_mask(self.bl.shape[0], self.bl.shape[1], center=center, radius=r)

        # Plot if wanted.
        if plotting is True:
            result[~roi] = 0
            plt.figure()
            plt.imshow(result)
            plt.show()

        return (roi)

    def create_circular_mask(self, height, weight, center=None, radius=None):
        '''
        :param height: height of an image.
        :param weihgt: weight of an image.
        :center: center of the circle mask.
        :radius: size of the radius the circle mask should have.
        :return: circle mask
        '''
        if center is None: # use the middle of the image
            center = [int(weight/2), int(height/2)]
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], weight - center[0], height - center[1])

        Y, X = np.ogrid[:height, :weight]

        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

        mask = dist_from_center <= radius
        return (mask)



#model = ChangeRegistrationModel()
#model.detectingChange()
