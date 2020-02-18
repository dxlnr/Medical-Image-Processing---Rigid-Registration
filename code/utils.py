import numpy as np
import cv2


def getPoints(outLiers):
    BLPoints = np.array([
							[180.69823,	115.30292],
							[144.02515,	1308.6024],
							[579.16498,	1331.1505],
                            ])
    FUPoints = np.array([
                        [401.0,	112.7],
							[314.7,	1314.8],
							[748.8,	1354.6],
							])
    if (outLiers == 'no_outliers'):
        pass
    elif (outLiers == 'with_outliers'):
        BLPointsOutLiers = np.array([[863.75000,	68.500000], [1018.2500,	302.50000],[812.75000,	928],[970.25000,	491.50000],[298.25000,	686.50000],[1324.2500,	1223.5000]])
        BLPoints= np.append(BLPoints, BLPointsOutLiers,axis=0)
        FUPointsOutLiers = np.array([[1105.2500,	95.500000],[1244.7500,	335.50000],[448.25000,	880],[475.25000,	472],[1367.7500,	773.50000],[275.75000,	1153]])
        FUPoints = np.append(FUPoints, FUPointsOutLiers,axis=0)
    else:
        print("Unknown string " + outLiers)
        exit()
    return BLPoints,FUPoints


def ransac(x, y, funcFindF, funcDist, minPtNum, iterNum, thDist, thInlrRatio):
    """
    Use RANdom SAmple Consensus to find a fit from X to Y.
    :param x: M*n matrix including n points with dim M
    :param y: N*n matrix including n points with dim N
    :param funcFindF: a function with interface f1 = funcFindF(x1,y1) where:
                x1 is M*n1
                y1 is N*n1 n1 >= minPtNum
                f is an estimated transformation between x1 and y1 - can be of any type
    :param funcDist: a function with interface d = funcDist(x1,y1,f) that uses f returned by funcFindF and returns the
                distance between <x1 transformed by f> and <y1>. d is 1*n1.
                For line fitting, it should calculate the distance between the line and the points [x1;y1];
                For homography, it should project x1 to y2 then calculate the dist between y1 and y2.
    :param minPtNum: the minimum number of points with whom can we find a fit. For line fitting, it's 2. For
                homography, it's 4.
    :param iterNum: number of iterations (== number of times we draw a random sample from the points
    :param thDist: inlier distance threshold.
    :param thInlrRatio: ROUND(THINLRRATIO*n) is the inlier number threshold
    :return: [f, inlierIdx] where: f is the fit and inlierIdx are the indices of inliers

    transalated from matlab by Adi Szeskin.
    """

    ptNum = len(x)
    thInlr = round(thInlrRatio*ptNum)


    inlrNum = np.zeros([iterNum,1])
    fLib= np.zeros(shape=(iterNum,3,3))
    for i in range(iterNum):
        permut = np.random.permutation(ptNum)
        sampleIdx = permut[range(minPtNum)]
        f1 = funcFindF(x[sampleIdx,:],y[sampleIdx,:])
        dist = funcDist(x,y,f1)
        b = dist<=thDist
        r = np.array(range(len(b)))
        inlier1 = r[b]
        inlrNum[i] = len(inlier1)
        if len(inlier1) < thInlr: continue
        fLib[i] = funcFindF(x[inlier1,:],y[inlier1,:])

    idx = inlrNum.tolist().index(max(inlrNum))
    f = fLib[idx]
    dist = funcDist(x,y,f);
    b = dist<=thDist
    r = np.array(range(len(b)))
    inlierIdx = r[b]
    return f, inlierIdx
