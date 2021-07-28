


from wand.image import Image
from PIL import Image as PI
import pytesseract
import argparse
import cv2
import os
import numpy as np
import io
from pdf2image import convert_from_path
import wx
import enchant
import wx
from enchant.checker import SpellChecker
from enchant.checker.wxSpellCheckerDialog import wxSpellCheckerDialog
from enchant.checker.CmdLineChecker import CmdLineChecker

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input image or pdf to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
ap.add_argument("-o", "--output", required=True,
    help="path to output the result")
ap.add_argument("-v", "--verbose", required=False,
    help="output detailed logs")
args = vars(ap.parse_args())

fn = args["input"]

if fn.endswith('f'):
    image_pdf = Image(filename= fn,resolution = 200)
    image_ini = image_pdf.convert('jpg')
    fn1 = fn[:len(fn)-4]
    path = ("%s.jpg" % (fn1))
    image_ini.save(filename=path)
    image_ini = cv2.imread(fn1+'.jpg')
else:
    image_ini = cv2.imread(fn)
  
    
    
# convert it to grayscale
gray = cv2.cvtColor(image_ini, cv2.COLOR_BGR2GRAY)
#gray = cv2.bitwise_not(gray)

# noise removal
#gray =  cv2.medianBlur(gray,5)
 
#thresholding
gray= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
#kernel = np.ones((5,5),np.uint8)
#gray= cv2.dilate(gray, kernel, iterations = 1)
    
#erosion
#gray = cv2.erode(gray, kernel, iterations = 1)

#opening - erosion followed by dilation
#gray =  cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

#canny edge detection
#gray = cv2.Canny(gray, 100, 200)

    

#def match_template(rotated, template):
#    return cv2.matchTemplate(rotated, template, cv2.TM_CCOEFF_NORMED) 

# Check if the picture needs to be rotated
coords = np.column_stack(np.where(gray > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make it positive
else:
	angle = -angle    
    
# rotate the image to deskew it
(h, w) = gray.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# write the grayscale image to disk as a temporary file so we canapply OCR to it
filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, rotated)

# load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file

config = "--psm 3"
text = pytesseract.image_to_string(filename, config=config)
print(text)

os.remove(filename)

# correct the spelling mistake

# Outout the result
text = text.encode('UTF-8')
f = open(args["output"],'wb')
f.write(text)
f.close()