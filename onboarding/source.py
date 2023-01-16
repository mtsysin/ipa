"""Simple onboarding code for"""
import numpy as np 
import cv2
import matplotlib.pyplot as plt

cv2.destroyAllWindows()
# Load sample image and perform simple operation
img = cv2.imread("test.jpeg", cv2.IMREAD_COLOR)

# Show image
def display(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)

# try converting to grayscale
# basic info
print(img.shape)
print(type(img))

# BT.709
# BT.601
# R, G, B
coefs_709 = [0.2126, 0.7152, 0.0722]
coefs_601 = [0.2126, 0.7152, 0.0722]
coefs_uniform = [0.33, 0.33, 0.33]

def convert_to_grayscale(img, coefs):
    coefs = np.array(coefs)
    return (img@coefs).astype(np.uint8)

img1 = convert_to_grayscale(img, coefs=coefs_709)
img2 = convert_to_grayscale(img, coefs=coefs_601)
img3 = convert_to_grayscale(img, coefs=coefs_uniform)

img_grayscale = img1

# print(img1.shape)
# print(img)

#Dispaly
# for img in [img1, img2, img3]:
#     display(img)

# try filtering:
# Test filters:
# Custom filter application for grayscale images:
# works only for widows with odd size
def apply_filter_zero_padding(filter, img):
    window_size = filter.shape[0]
    padding = window_size//2
    img_size = img.shape
    # Pad the image:
    corner_padding = np.zeros((padding, padding))
    horizontal_padding = np.zeros((padding, img_size[1]))
    vertical_padding = np.zeros((img_size[0], padding))

    img = np.concatenate([
            np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
            np.concatenate([vertical_padding, img, vertical_padding], axis=1),
            np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
        ],
        axis=0
    )

    img_filt = np.zeros(img_size)
    # apply filter
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            img_filt[i][j] = np.sum(img[i:i+window_size,j:j+window_size] * filter)
    
    return img_filt.astype(np.uint8)
    # print(img)

# Show image histogram
def img_histogram(img, bin_size):
    hist = [0] * (256//bin_size)
    for row in img:
        for pixel in row:
            hist[pixel//bin_size] += 1
    plt.bar([i*bin_size for i in range(len(hist))], hist)
    plt.show()

def invert_img(img):
    return 255 - img
    

blur_filter = np.array(
    [[1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25]]
)

# blurred_img = apply_filter_zero_padding(filter=blur_filter, img=img_grayscale)

# display(blurred_img)
# img_histogram(img_grayscale, 2)
# display(invert_img(img_grayscale))

blur_filter = np.array(
    [[1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25]]
)

cv2.destroyAllWindows()

