from utils import get_ocr

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import pandas as pd
import pytesseract
import cv2
from pytesseract import Output
import imageio



im = imageio.imread('./images/recepeit3.jpg')
df = get_ocr(im, plot_result=True)
# Keep only the text
text = list(df['text'])

# Save it to txt file
# with open('tesseract_text.txt', 'w') as output:
#     for word in text:
#         output.write("%s" % word)
#     output.close()

print("\n\n TEXT: \n\n", ' '.join(text))