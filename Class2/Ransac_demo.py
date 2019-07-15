import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img0 = cv2.imread(r'D:\Lynn\code\ai-for-cv-course-4th\Class2\object2.jpg')
    img1 = cv2.imread(r"D:\Lynn\code\ai-for-cv-course-4th\Class2\sence2.jpg")
    # create sift class
    sift = cv2.xfeatures2d.SIFT_create()
    # compute SIFT descriptor
    kp0,des0 = sift.detectAndCompute(img0,None)
    kp1,des1 = sift.detectAndCompute(img1,None)
    img0_sift = cv2.drawKeypoints(img0,kp0,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img1_sift = cv2.drawKeypoints(img1,kp1,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('scene', img0_sift)
    cv2.imshow('object', img1_sift)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KATREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KATREE, tree = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des0, des1, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    good = [m[0] for m in good]
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w,d = img0.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask =None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img0,kp0,img1,kp1,good,None,**draw_params)
    #plt.imshow(img3, 'gray'),plt.show()
    cv2.imshow('Match', img3)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()