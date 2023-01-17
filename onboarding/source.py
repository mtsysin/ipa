"""Simple onboarding code for"""
import numpy as np 
import cv2
import matplotlib.pyplot as plt

cv2.destroyAllWindows()

# BT.709
# BT.601
# R, G, B
COEFS_709 = [0.2126, 0.7152, 0.0722]
COEFS_601 = [0.2126, 0.7152, 0.0722]
COEFS_UNIFORM = [0.33, 0.33, 0.33]
BLUR_FILT = np.array(
    [[1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25],
    [1/25, 1/25, 1/25, 1/25, 1/25]]
)

# SOBEL_X = np.array(
#     [[2, 2, 4, 2, 2],
#     [1, 1, 2, 1, 1],
#     [0, 0, 0, 0, 0],
#     [-1, -1, -2, -1, -1],
#     [-2, -2, -4, -2, -2]]
# )

# SOBEL_Y = np.array(
#     [[2, 1, 0, -1, -2],
#     [2, 1, 0, -1, -2],
#     [4, 2, 0, -2, -4],
#     [2, 1, 0, -1, -2],
#     [2, 1, 0, -1, -2]]
# )

SOBEL_X = np.array(
    [[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]
)

SOBEL_Y = np.array(
    [[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]
)

# print(img1.shape)
# print(img)

#Dispaly
# for img in [img1, img2, img3]:
#     display(img)

# try filtering:
# Test filters:
# Custom filter application for grayscale images:
# works only for widows with odd size
class Image:
    def __init__(self, path = None, img = None) -> None:
        # Load sample image to perform simple operations
        if path:
            self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            self.img = img
        self.size = self.img.shape

    def display(self):
        # Show image
        cv2.imshow("image", self.img)
        cv2.waitKey(0)

    def convert_to_grayscale(self, coefs):
        coefs = np.array(coefs)
        return Image(img = (self.img@coefs).astype(np.uint8))

    def apply_filter_zero_padding(self, filter):
        window_size = filter.shape[0]
        padding = window_size//2
        img_size = self.size
        # Pad the image:
        corner_padding = np.zeros((padding, padding))
        horizontal_padding = np.zeros((padding, img_size[1]))
        vertical_padding = np.zeros((img_size[0], padding))

        img_temp = np.concatenate([
                np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
                np.concatenate([vertical_padding, self.img, vertical_padding], axis=1),
                np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
            ],
            axis=0
        )

        img_filt = np.zeros(img_size)
        # apply filter
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                img_filt[i][j] = np.sum(img_temp[i:i+window_size,j:j+window_size] * filter)
        
        return Image(img = img_filt.astype(np.uint8))

    # Show image histogram
    def show_histogram(self, bin_size):
        hist = [0] * (256//bin_size)
        for row in self.img:
            for pixel in row:
                hist[pixel//bin_size] += 1
        plt.bar([i*bin_size for i in range(len(hist))], hist)
        plt.show()

    def invert(self):
        return Image(img = 255 - self.img)

    def threshold(self, threshold):
        return Image(img = (self.img > threshold).astype(np.uint8)*255)

    def sobel_edge(self):
        x = self.apply_filter_zero_padding(SOBEL_X)
        y = self.apply_filter_zero_padding(SOBEL_Y)
        return Image(img = np.sqrt(np.square(x.img) + np.square(y.img)).astype(np.uint8))
    
image = Image(path="test.jpeg")

# Compare grayscale conversion
# for coefs in [COEFS_601, COEFS_709, COEFS_UNIFORM]:
#     image.convert_to_grayscale(coefs).display()

image = image.convert_to_grayscale(COEFS_709)

# Try blurred image
blurred_image = image.apply_filter_zero_padding(filter=BLUR_FILT)
# blurred_image.display()

# Try to show image histogram
# image.show_histogram(2)

# Invert image
# image.invert().display()

# Threhold image
# image.threshold(100).display()

# Edge detection
# image.sobel_edge().display()
i = image.apply_filter_zero_padding(filter=SOBEL_X)
i.display()

i = image.apply_filter_zero_padding(filter=SOBEL_Y)
i.display()




cv2.destroyAllWindows()

