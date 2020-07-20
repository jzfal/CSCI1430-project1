# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape) # pad the border with zeros

    # inspiration for code coming from https://www.allaboutcircuits.com/technical-articles/two-dimensional-convolution-in-image-processing/

    ##################
    # Your code here #
    # check for even dimension 
    # print([[x%2==0 for x in image.shape],[x%2==0 for x in kernel.shape]])
    if any([x%2==0 for x in image.shape] + [x%2==0 for x in kernel.shape]):
        raise Exception("Dimensions cannot be even, np.shape must return non-even shape")
    else:
        # no need to pad
        # run imfilter
        # will resize the image according to the kernal/mask
        # change the inputs to raster matrices
        new_kernel = np.flip(kernel) # flip the kernel 
        # need to zero pad the input image
        # the formula for padding is the number of rows or columns to be zero padded on each side of the input image is given by (number of rows or columns in the kernel-1)

        row_pads = new_kernel.shape[0]-1
        col_pads = new_kernel.shape[1]-1
        
        if len(image.shape) == 3: # for color images
            image_padded = np.zeros(shape = (row_pads*2 + image.shape[0], col_pads*2 + image.shape[1], 3))
            if col_pads == 0 and row_pads == 0:
                image_padded[:,:,:] = image
            elif col_pads == 0:
                image_padded[row_pads:-row_pads,:,:] = image
            elif row_pads == 0:
                image_padded[:,col_pads:-col_pads,:] = image
            else:
                image_padded[row_pads:-row_pads,col_pads:-col_pads,:] = image
        else:
            if col_pads == 0 and row_pads == 0:
                image_padded[:,:] = image
            elif col_pads == 0:
                image_padded[row_pads:-row_pads,:] = image
            elif row_pads == 0:
                image_padded[:,col_pads:-col_pads] = image
            else:
                image_padded[row_pads:-row_pads,col_pads:-col_pads] = image
        # plt.imshow(image_padded)
        # print(image == image_padded)
        # plt.imshow(image_padded)
        # plt.show()
        # return




        if len(image.shape) == 3: # for color images # can do a break if rank is 2
            for i in range(3):# for rgb
                # the pads depend on the size of the kernel
                # the size of the border depends on the kernel size, if 3 == 1, 5 == 2 and so on, number of rows indicate the top and bottom border, number of cols indicate the left and right border
                # num_row_iter = image.shape[0] - new_kernel.shape[0]
                for j in range(filtered_image.shape[0]): # this is the range over the rows (number of rows in output matrix)
                    # num_col_iter = image.shape[1] - new_kernel.shape[1]
                    for k in range(filtered_image.shape[1]):
                        # need to find the starting point for the col and row
                        # new_kernel[j + num_col_iter/2]

                        filtered_image[j,k,i] = np.sum(image_padded[j:j+row_pads+1, k:k+col_pads+1, i] * new_kernel) # dot product
        else:
            for j in range(filtered_image.shape[0]): # this is the range over the rows (number of rows in output matrix)
                # num_col_iter = image.shape[1] - new_kernel.shape[1]
                for k in range(filtered_image.shape[1]):
                    # need to find the starting point for the col and row
                    # new_kernel[j + num_col_iter/2]
                    filtered_image[j,k] = np.sum(image_padded[j:j+row_pads+1, k:k+col_pads+1] * new_kernel) # dot product

                    





    # print('my_imfilter function in student.py needs to be implemented')
    ##################

    return filtered_image

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    low_frequencies = np.zeros(image1.shape) # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = np.zeros(image1.shape) # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = np.zeros(image1.shape) # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.

    return low_frequencies, high_frequencies, hybrid_image
