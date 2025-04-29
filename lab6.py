"""
    Jesse Han
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Lab 6
    Resources: None
"""
from PIL import Image
import numpy as np
import math

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
            x_to = (-(filter_width - 1) // -2) + 1
            y_from = (filter_height - 1) // -2
            y_to = (-(filter_height - 1) // -2) + 1
            for x in range(x_from, x_to):
                for y in range(y_from, y_to):
                    if i+x >= 0 and i+x < width and j+y >= 0 and j+y < height:
                        # Reversing the offsets since the filters are indexed from 0, not the negative offset
                        Gx += pxl[i+x,j+y] * filterX[-y_from + y][-x_from + x]
                        Gy += pxl[i+x,j+y] * filterY[-y_from + y][-x_from + x]
            G = (int) (math.sqrt(Gx ** 2 + Gy ** 2))
            output_pixels[i,j] = G
    return output

image_name = 'dog.png'
image = Image.open(image_name).convert('L')
roberts = apply_edge_convolution(image, robertsX, robertsY)
prewitt = apply_edge_convolution(image, prewittX, prewittY)
sobel = apply_edge_convolution(image, sobelX, sobelY)

roberts.save(f"roberts_{image_name}")
prewitt.save(f"prewitt_{image_name}")
sobel.save(f"sobel_{image_name}")
