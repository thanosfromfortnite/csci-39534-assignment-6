"""
    Jesse Han
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Lab 6
    Resources: Scikit-image Unsharp masking
               https://scikit-image.org/docs/0.24.x/auto_examples/filters/plot_unsharp_mask.html
               Wikipedia Unsharp masking
               https://en.wikipedia.org/wiki/Unsharp_masking
"""
from PIL import Image
import numpy as np
import math
import statistics

# Defining filters
robertsX = (
    (1, 0),
    (0, -1)
)
robertsY = (
    (0, 1),
    (-1, 0)
)
prewittX = (
    (-1, 0, 1),
    (-1, 0, 1),
    (-1, 0, 1)
)
prewittY = (
    (1, 1, 1),
    (0, 0, 0),
    (-1, -1, -1)
)
sobelX = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1)
)
sobelY = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1)
)

def apply_edge_convolution(image, filterX, filterY):
    img = image.copy()
    pxl = img.load()
    width = img.size[0]
    height = img.size[1]
    filter_width = len(filterX)
    filter_height = len(filterX[0])
    
    output = Image.new(mode='L', size=(width, height))
    output_pixels = output.load()
    
    for i in range(width):
        for j in range(height):
            Gx = 0
            Gy = 0
            # Range values are centered around [i,j] based on filter size
            # Creating offsets based on index sizes
            x_from = (filter_width - 1) // -2
            x_to = (filter_width - 1) // 2 + 1
            y_from = (filter_height - 1) // -2
            y_to = (filter_height - 1) // 2 + 1
            for x in range(x_from, x_to):
                for y in range(y_from, y_to):
                    if i+x >= 0 and i+x < width and j+y >= 0 and j+y < height:
                        # Reversing the offsets since the filters are indexed from 0, not the negative offset
                        Gx += pxl[i+x,j+y] * filterX[-y_from + y][-x_from + x]
                        Gy += pxl[i+x,j+y] * filterY[-y_from + y][-x_from + x]
            G = (int) (math.sqrt(Gx ** 2 + Gy ** 2))
            output_pixels[i,j] = G
    img.close()
    return output

def apply_median_filter(image, size=3):
    img = image.copy()
    pxl = img.load()
    width = img.size[0]
    height = img.size[1]

    output = Image.new(mode='L', size=(width,height))
    output_pixels = output.load()

    for i in range(width):
        for j in range(height):
            neighbors = []
            for x in range((size-1)//-2, (size-1)//2+1):
                for y in range((size-1)//-2, (size-1)//2+1):
                    if i+x >= 0 and i+x < width and j+y >= 0 and j+y < height:
                        neighbors.append(pxl[i+x,j+y])
            output_pixels[i,j] = int(statistics.median(neighbors))
    img.close()
    return output


gaussian_filter = (
    (1.0/16.0, 2.0/16.0, 1.0/16.0),
    (2.0/16.0, 4.0/16.0, 2.0/16.0),
    (1.0/16.0, 2.0/16.0, 1.0/16.0)
)
def apply_other_filter(image, image_filter):
    img = image.copy()
    pxl = img.load()
    width = img.size[0]
    height = img.size[1]

    filter_size = len(image_filter)

    output = Image.new(mode='L', size=(width,height))
    output_pixels = output.load()

    for i in range(width):
        for j in range(height):
            dividend = 0.0
            for x in range((filter_size-1)//-2, (filter_size-1)//2+1):
                for y in range((filter_size-1)//-2, (filter_size-1)//2+1):
                    if i+x >= 0 and i+x < width and j+y >= 0 and j+y < height:
                        dividend += pxl[i+x,j+y] * image_filter[y + (filter_size-1)//2][x + (filter_size-1)//2]
            output_pixels[i,j] = (int) (dividend)
    img.close()
    return output

def unsharp_masking(original, blurred, weight=1.0):
    output = original.copy()
    output_pixels = output.load()
    blurred_pixels = blurred.load()
    width = output.size[0]
    height = output.size[1]

    for i in range(width):
        for j in range(height):
            output_pixels[i,j] = output_pixels[i,j] + (int) (weight * (output_pixels[i,j] - blurred_pixels[i,j]))
    return output

'''
dog_name = 'dog.png'
dog = Image.open(dog_name).convert('L')
dog.save(f"grayscale_{dog_name}")
roberts_dog = apply_edge_convolution(dog, robertsX, robertsY)
prewitt_dog = apply_edge_convolution(dog, prewittX, prewittY)
sobel_dog = apply_edge_convolution(dog, sobelX, sobelY)

roberts_dog.save(f"roberts_{dog_name}")
prewitt_dog.save(f"prewitt_{dog_name}")
sobel_dog.save(f"sobel_{dog_name}")
'''

'''
snow_name = 'snow.png'
snow = Image.open(snow_name).convert('L')
snow.save(f"grayscale_{snow_name}")
sobel_snow = apply_edge_convolution(snow, sobelX, sobelY)
sobel_snow.save(f"sobel_{snow_name}")

snow_5x5 = apply_median_filter(snow, 5)
snow_7x7 = apply_median_filter(snow, 7)

snow_5x5.save(f"5x5_{snow_name}")
snow_7x7.save(f"7x7_{snow_name}")

sobel_snow_5x5 = apply_edge_convolution(snow_5x5, sobelX, sobelY)
sobel_snow_7x7 = apply_edge_convolution(snow_7x7, sobelX, sobelY)

sobel_snow_5x5.save(f"5x5_sobel_{snow_name}")
sobel_snow_7x7.save(f"7x7_sobel_{snow_name}")
'''

droplets_name = 'droplets.png'
droplets = Image.open(droplets_name).convert('L')
droplets.save(f"grayscale_{droplets_name}")
sobel_droplets = apply_edge_convolution(droplets, sobelX, sobelY)
sobel_droplets.save(f"sobel_{droplets_name}")

droplets_5x5 = apply_median_filter(droplets, 5)
droplets_gaussian = apply_other_filter(droplets, gaussian_filter)
droplets_7x7_gaussian = apply_other_filter(apply_median_filter(droplets, 7), gaussian_filter)

droplets_5x5.save(f"5x5_{droplets_name}")
droplets_gaussian.save(f"gaussian_{droplets_name}")
droplets_7x7_gaussian.save(f"7x7_gaussian_{droplets_name}")

sobel_droplets_5x5 = apply_edge_convolution(droplets_5x5, sobelX, sobelY)
sobel_droplets_gaussian = apply_edge_convolution(droplets_gaussian, sobelX, sobelY)
sobel_droplets_7x7_gaussian = apply_edge_convolution(droplets_7x7_gaussian, sobelX, sobelY)

sobel_droplets_5x5.save(f"5x5_sobel_{droplets_name}")
sobel_droplets_gaussian.save(f"gaussian_sobel_{droplets_name}")
sobel_droplets_7x7_gaussian.save(f"7x7_gaussian_sobel_{droplets_name}")

unsharp = unsharp_masking(droplets, droplets_7x7_gaussian, -1.0)
unsharp.save(f"unsharp_{droplets_name}")

unsharp = apply_edge_convolution(unsharp, sobelX, sobelY)
unsharp.save(f"unsharp_sobel_{droplets_name}")
