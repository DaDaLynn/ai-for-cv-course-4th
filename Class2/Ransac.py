import random

def computeHomography(inliers):
    return homography

def EuclideanDistance(A, B):
    return dis

def ransacMatching(A, B):
    #step 1 : Choose 4 inliers randomly
    random.shuffle(A)
    inliers = list(zip(A[:4], B[:4]))
    outliers = A[4:-1]
    k = 100
    for iter in range(k):
        #step 2 : Get the homography of the inliers
        homography0 = computeHomography(inliers)
        #sep 3 : test all the other outliers
        threshold = 0.01
        for i in outliers:
            proj = hogography * i
            for j in B:
                if EuclideanDistance(i, j) < threshold:
                    inliers.append([i, j])
                    break
        homography1 = computeHomography(inliers)
        if homography0 == homography1:
            return homography0
        for i in inliers:
            if i[0] in outliers:
                outliers.remove(i[0])
        
        





    