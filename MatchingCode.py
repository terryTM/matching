import cv2
import numpy as np
import copy
import time
import sys
import os

"""
# load image
img = cv2.imread("/Users/terryma/Documents/emshot1.png")

# convert to graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image as mask
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

# negate mask
mask = 255 - mask

# apply morphology to remove isolated extraneous noise
# use borderconstant of black since foreground touches the edges
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# anti-alias the mask -- blur then stretch
# blur alpha channel
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

# linear stretch so that 127.5 goes to 0, but 255 stays 255
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# put mask into alpha channel
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# display result, though it won't show transparency
cv2.imshow("MASK", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
#from rembg.bg import remove




#image = PIL.Image.open("/Users/terryma/Downloads/E11_mouse_embryo_bright_field_images.png")


#image = Image.open("/Users/terryma/Downloads/whole_embryo_H3K4me3_E11_50um.png")
#image_array = np.array(image)


#print(image_in)


#image2 = Image.open("/Users/terryma/Downloads/whole_embryo_H3K27me3_E11_50um.png")
#image_array2 = np.array(image2)
#image_in2 = cv2.cvtColor(cv2.imread("/Users/terryma/Downloads/whole_embryo_H3K27me3_E11_50um.png"), cv2.COLOR_BGR2GRAY)


#zero_arr = np.zeros((image_array.shape[0], image_array.shape[1], 4))



#image_in = cv2.imread("/Users/terryma/Downloads/whole_embryo_H3K4me3_E11_50um.png", 0)

def edge(path, color_in):


    image_in2 = cv2.imread(path)

    image_in = cv2.cvtColor(image_in2, cv2.COLOR_RGB2GRAY)
    print(image_in.shape)


    #print(zero_arr.shape)
    image_in = cv2.GaussianBlur(image_in, (5,5), 0)
    #image_in = cv2.medianBlur(image_in,5)

    """
    for i in range(image_in.shape[0]):
        for j in range(image_in.shape[1]):
            
            if (image_in[i][j] >= 183):
                image_in[i][j] = 255
            if (image_in[i][j] < 183):
                image_in[i][j] = 0

    for i in range(image_in2.shape[0]):
        for j in range(image_in2.shape[1]):
            
            if (image_in2[i][j] > 180):
                image_in2[i][j] = 255
            if (image_in2[i][j] < 177):
                image_in2[i][j] = 0
    """

    gray = image_in


    #original = image.copy()
    mask = np.zeros(image_in.shape, dtype=np.uint8)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,3)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,25,5)

    cv2.imshow('thresholded', thresh)
    cv2.waitKey()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    a = np.zeros(image_in.shape, np.uint8)

    cim = np.zeros_like(a)
    cv2.drawContours(cim, contours, -1, 255, 1)
    cim = cv2.medianBlur(cim,3)

    close = cim

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    close = cv2.morphologyEx(close, cv2.MORPH_BLACKHAT, kernel)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    #close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)
    #close = cv2.medianBlur(close, 17)

    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    #print(hierarchy.shape)



    new_alpha = np.zeros_like(close)
    new_alpha = cv2.drawContours(new_alpha, [big_contour], 0, 255, -1)

    #print(contours)
    #holes = [contours[i] for i in range(len(contours)) if hierarchy[i][3] >= 0]

    out = np.zeros_like(close)

    cv2.drawContours(out, contours, -1, 255, 3)



    def undesired_objects(image):
        image = image.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        cv2.imshow("Biggest component", img2)
        cv2.waitKey()

    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(out, connectivity=4)
    #sizes = stats[:, -1]

    """
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255

    print(img2.shape)
    """

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    #out = cv2.morphologyEx(out, cv2.MORPH_BLACKHAT, kernel)



    #c = max(contours, key = cv2.contourArea)
    #x,y,w,h = cv2.boundingRect(c)
    #cv2.rectangle(close,(x,y),(x+w,y+h),(0,255,0),2)

    #out = undesired_objects(out)

    new_alpha = cv2.medianBlur(new_alpha, 5)


    edges = cv2.Canny(new_alpha, 100, 200)
    indices = np.where(edges != [0])
    coordinates = zip(indices[0], indices[1])
    print(list(coordinates))

    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    rgb *= np.array(color_in,np.uint8)
    out = np.bitwise_or(image_in2, rgb)
    print(out.shape)


    cv2.imshow('edge', rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return rgb



    """from scipy import ndimage

    def sobel_filters(img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        return (G, theta)"""


    """
    #edges = cv2.Canny(image_in,50,100)
    img = np.invert(image_in)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    img = cv2.ximgproc.thinning(img)

    #kernel = np.ones((15, 15), np.uint8)
    #closing = cv2.morphologyEx(image_in, cv2.MORPH_CLOSE, kernel)

    #edges = cv2.Canny(closing, 100, 200)


    plt.imshow(img, cmap='gray')
    plt.show()
    """


    """
    print(image_array.shape)
    print(image_array2.shape)

    figure, ax = plt.subplots(1, 2)

    plt.rcParams['figure.figsize']=(20,15)


    image_in2=np.pad(image_in2, ((26,26), (169,168)), 'maximum')
    fin_1 = image_in2 + image_in
    ax[0].imshow(image_in, cmap='gray')
    ax[0].axis('off')

    ax[1].imshow(image_in2, cmap=plt.cm.gray)
    ax[1].axis('off')


    plt.show()
    """

firstpath = sys.argv[1]
secondpath = sys.argv[2]
if os.path.exists(firstpath) and os.path.exists(secondpath):
    im1 = edge(firstpath, (0, 0, 1))
    im2 = edge(secondpath, (1, 0, 0))

import math

img1 = im1
img2 = im2

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#from simpleicp import PointCloud, SimpleICP


#pc_fix = PointCloud(im1_gray, columns=["x", "y", "z"])
#pc_mov = PointCloud(im2_gray, columns=["x", "y", "z"])

#icp = SimpleICP()
#icp.add_point_clouds(pc_fix, pc_mov)
#H, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)

#print(H)

"""
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3),plt.show()
"""

# Convert images to grayscale for computing the rotation via ECC method


# Find size of image1
sz = im1.shape

# Define the motion model - euclidean is rigid (SRT)
warp_mode = cv2.MOTION_HOMOGRAPHY

# Define 2x3 matrix and initialize the matrix to identity matrix I (eye)
warp_matrix = np.eye(3, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 5000

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-3

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

# Warp im2 using affine
im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# write output
cv2.imshow('outfile', im2_aligned+im1)
cv2.waitKey()
cv2.destroyAllWindows()
# Print rotation angle
row1_col0 = warp_matrix[0,1]
angle = math.degrees(math.asin(row1_col0))
print(angle)
