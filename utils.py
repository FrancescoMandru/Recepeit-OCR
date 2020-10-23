import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import pandas as pd
import pytesseract
import cv2
from pytesseract import Output
import imageio

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Unsharp masking
def unsharp_mask(image, k=1):
    
    gaussian = cv2.GaussianBlur(image, (7, 7), 3)
    unsharp_image = cv2.addWeighted(image, k+1, gaussian, -k, 0)
    
    return unsharp_image

# Resize image
def resize_img(image, scale):
    
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    
    return resized

# Dilation of the image
def dilate(image, kernel_size=5):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)


def get_ocr(screenshot, plot_result=False, conf=60):

    '''This function does a first preprocessing the image
    and then applies Tesseract. You can select the level of confidence 
    and also to plot the bounding boxes 
    
    Returns a pandas dataframe with many info: box_left = Bounding box 
    coordinates text etc.." 
    '''
    
    page_screenshot = resize_img(screenshot, scale=100)
    gray_img = get_grayscale(page_screenshot)
    th = thresholding(gray_img)
    sharped_img = unsharp_mask(th)
    # Call pytesseract after the preprocessing of the image
    txt = pytesseract.image_to_data(sharped_img, output_type='data.frame')
    # Remove everything that is not text from tesseract result
    txt = txt[txt['text'].notna()]
    # Keep only elements with a confidence higher than confidence 
    txt = txt.loc[txt['conf'] >= conf]
    # Plot the result
    if(plot_result):
        get_bbox_img(sharped_img, conf)

    # Return a pandas dataframae with
    
    return txt

def get_bbox_img(processed_img, conf):
    # Get the bounding box with the following parameters (they work well and are the best)
    custom_oem_psm_config = r'--oem 3 --psm 3'
    d = pytesseract.image_to_data(processed_img, config=custom_oem_psm_config, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes-1):
        
        if( int(d['conf'][i]) > conf and int(d['level'][i]) == 5 and len(d['text'][i].strip()) > 0):
            #print('Text: {} length: {}'.format(str(d['text'][i]),len(d['text'][i])))
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.figure(figsize=(10,10))
    plt.imshow(processed_img, cmap='gray')
    plt.show()