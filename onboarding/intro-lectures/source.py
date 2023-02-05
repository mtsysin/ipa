"""Simple onboarding code for"""
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from enum import Enum, auto

class Norm(Enum):
    """Helper type"""
    MinMax = auto()
    Max = auto()
    Clip = auto()

cv2.destroyAllWindows()

# BT.709
# BT.601
# R, G, B
COEFS_709 = [0.2126, 0.7152, 0.0722]
COEFS_601 = [0.2126, 0.7152, 0.0722]
COEFS_UNIFORM = [0.33, 0.33, 0.33]

def create_blur_filt(size):
    """Generaes box blur windiws, just 1/n^2 in each cell"""
    return np.array([[1/size**2 for _ in range(size)] for _ in range(size)]).astype(np.float32)

SOBEL_X = np.array(
    [[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]
).astype(np.float32)

SOBEL_Y = np.array(
    [[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]
).astype(np.float32)

UNSHARP_3 = np.array(
    [[0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]]
).astype(np.float32)

# print(img1.shape)
# print(img)

#Dispaly
# for img in [img1, img2, img3]:
#     display(img)

def normalize(img):
    """Normalizes an image scaling maximum to 1"""
    img.astype(np.float32)
    return img / np.max(img), np.max(img)


class Image:
    def __init__(self, path = None, img = None) -> None:
        # Load sample image to perform simple operations
        if path:
            self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            self.img = img
        self.size = self.img.shape

    def display(self):
        """Displays image with cv2 module"""
        cv2.imshow("image", self.img)
        cv2.waitKey(0)


    def convert_to_grayscale(self, coefs):
        """Convert image to grayscale with given conversion coefficients"""
        coefs = np.array(coefs)
        return Image(img = (self.img@coefs).astype(np.uint8))

    def median_filter(self, window_size):
        """Applies a median filter to an image"""
        pad = window_size//2
        img_filt = np.zeros(self.size).astype(np.float32)
        # Apply filter
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Take a median of the window
                img_filt[i][j] = np.median(self.img[min(i-pad, i) : max(i+pad, i)+1, min(j-pad, j) : max(j+pad, j)+1])
        return Image(img = img_filt.astype(np.uint8))


    def apply_filter_zero_padding(self, filter, norm = Norm.Max, return_raw = False):
        """filtering image with zero padding"""
        window_size = filter.shape[0]
        padding = window_size//2
        img_size = self.size

        # Pad the image, create block of zeros:
        corner_padding = np.zeros((padding, padding))
        horizontal_padding = np.zeros((padding, img_size[1]))
        vertical_padding = np.zeros((img_size[0], padding))

        # Put all the blocks together with the original image in the middle.
        img_temp = np.concatenate([
                np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
                np.concatenate([vertical_padding, self.img, vertical_padding], axis=1),
                np.concatenate([corner_padding, horizontal_padding, corner_padding], axis=1),
            ],
            axis=0
        )

        # Normalize image
        img_temp, norm_factor = normalize(img_temp)

        # Fill each pixel of the filtered version with the sum of element-vise products.
        img_filt = np.zeros(img_size).astype(np.float32)
        # apply filter
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                img_filt[i][j] = np.sum(img_temp[i:i+window_size,j:j+window_size] * filter)

        if return_raw:
            return img_filt

        # Choose how to normalize image
        if norm == Norm.MinMax:
            # Scale for minimum and maximum
            img_filt = ((img_filt - np.min(img_filt)) / (np.max(img_filt) - np.min(img_filt))) * 255
        elif norm == Norm.Max:
            # Scale only for maximum (minimum == 0)
            img_filt = img_filt / np.max(img_filt) * 255
        elif norm == Norm.Clip:
            # Clip image with 0 and 255
            img_filt = np.clip(img_filt * norm_factor, a_max=255, a_min=0)
            # print(img_filt)
        

        
        return Image(img = img_filt.astype(np.uint8))

    # Show image histogram
    def show_histogram(self, bin_size):
        """Shows histogram of a grayscale image"""
        hist = [0] * (256//bin_size)    # Initialize
        for row in self.img:    
            for pixel in row:   # Run loop and add 1 to the corresponding bin
                hist[pixel//bin_size] += 1
        plt.bar([i*bin_size for i in range(len(hist))], hist)
        plt.show()

    def invert(self):
        """Invert an image"""
        return Image(img = 255 - self.img)

    def threshold(self, threshold):
        """Threshold image with given threshold value"""
        return Image(img = (self.img > threshold).astype(np.uint8)*255) # Either 0 or 255

    def sobel_edge(self):
        """Performs sobel edge detection"""
        x = self.apply_filter_zero_padding(SOBEL_X, return_raw=True) # X-derivative
        y = self.apply_filter_zero_padding(SOBEL_Y, return_raw=True) # Y-derivative
        res = np.sqrt((np.square(x) + np.square(y))/2)  # Norm of the derivative
        res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255 # Renormalize to appropriate values
        return Image(img = res.astype(np.uint8))
    
# Testing code

image = Image(path="test.jpeg")

# Compare grayscale conversion
# for coefs in [COEFS_601, COEFS_709, COEFS_UNIFORM]:
#     image.convert_to_grayscale(coefs).display()

image = image.convert_to_grayscale(COEFS_709)
# image.display()
# image.show_histogram(2)


# Try blurred image
blurred_image = image.apply_filter_zero_padding(filter=create_blur_filt(3))
blurred_image.display()
# blurred_image.show_histogram(2)

# Try to show image histogram
# image.show_histogram(2)

# Invert image
# image.invert().display()

# Threhold image
# image.threshold(100).display()

# Edge detection
# Show intermediate steps:
# image.apply_filter_zero_padding(SOBEL_X, norm=Norm.MinMax).display()
# image.apply_filter_zero_padding(SOBEL_Y, norm=Norm.MinMax).display()
# edges = image.sobel_edge()
# edges.display()
# edges.show_histogram(2)
# edges.threshold(30).display()
# i = image.apply_filter_zero_padding(filter=SOBEL_X)
# i.display()

# Unsharp masking:
blurred_image_unsharp = blurred_image.apply_filter_zero_padding(UNSHARP_3, norm=Norm.Clip)
blurred_image_unsharp.display()
# blurred_image.apply_filter_zero_padding(UNSHARP, norm=Norm.Clip).show_histogram(2)

# Median filter:
# blurred_image_unsharp.median_filter(5).display()


cv2.destroyAllWindows()

