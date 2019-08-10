#import libraries 
import pandas as pd 
import numpy as np
import os 
pdf_file_path = 'leac203.pdf'
import PyPDF2 as pyPdf
from subprocess import call
import cv2 
from tabula import read_pdf

image_source_dir = 'D:/Data_Science_Work/table_detection/png'
image_target_dir = 'D:/Data_Science_Work/table_detection/png_preprocessed'
if not os.path.exists(image_source_dir):
    call('gswin32 -dBATCH -dNOPAUSE -sDEVICE=png16m -dGraphicsAlphaBits=4 \
    -dTextAlphaBits=4 -r600 -sOutputFile="D:/Data_Science_Work/table_detection/png/page_%d.png" \
    D:/Data_Science_Work/table_detection/leac203.pdf')

def preprocessimage(image):
    img = cv2.imread(os.path.join(image_source_dir, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
    
    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))
    target_file = os.path.join(image_target_dir, image)
    #print("Writing target file {}".format(target_file))
    cv2.imwrite(target_file, transformed_image)
    

reader = pyPdf.PdfFileReader(open(pdf_file_path, mode='rb' ))
num_of_pages = reader.getNumPages() 
print(num_of_pages)

img = cv2.imread('D:/Data_Science_Work/table_detection/png/page_9.png')
x0,y0,x2,y2 = 925, 2817, 3895, 5068
img = cv2.rectangle(img,(x0,y0),(x2,y2),(255,0,0),2)
img = cv2.resize(img, (600, 800)) 
cv2.imshow('Window',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for page_num in range(num_of_pages):
    page = 'page_' + str(page_num + 1) + '.png'
    preprocessimage(page)
    bounding_box = call('lumi predict --checkpoint 336265de5d17 D:/Data_Science_Work/table_detection/png/' + page)
    t = [(a )/8 for a in bounding_box]
    df = read_pdf(pdf_file, encoding = 'utf-8', pages = page_num, stream = True, 
              multiple_tables = True, area = t)

'''
with(Image(filename=diag,resolution=200)) as source:
    images=source.sequence
    pages=len(images)
    for i in range(pages):
        Image(images[i]).save(filename=str(i)+'.png')
'''       

from PIL import Image
from tesserocr import PyTessBaseAPI, PSM

with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
    image = Image.open("/usr/src/tesseract/testing/eurotext.tif")
    api.SetImage(image)
    api.Recognize()

    it = api.AnalyseLayout()
    orientation, direction, order, deskew_angle = it.Orientation()
    print("Orientation: {:d}".format(orientation))
    print("WritingDirection: {:d}".format(direction))
    print("TextlineOrder: {:d}".format(order))
    print("Deskew angle: {:.4f}".format(deskew_angle))
    
from tesserocr import PyTessBaseAPI, RIL, iterate_level, PT , OEM, PSM
from PIL import Image, ImageDraw
import os
tess_path = r"C:\Program Files\Tesseract-OCR\tessdata"
image = r"E:\Data_Science\data\images\gln20070618016-12.png"
img = Image.open(image)
draw = ImageDraw.Draw(img)
with PyTessBaseAPI(path = tess_path, oem = OEM.TESSERACT_LSTM_COMBINED) as api:
    api.SetImageFile(image)
    api.SetVariable("textord_tabfind_find_tables", "T")
    #api.SetVariable("textord_tablefind_recognize_tables", "T")
    api.Recognize()
    ri = api.GetIterator()
    level = RIL.BLOCK
    for r in iterate_level(ri, level):
        #print(r.BlockType())
        if r.BlockType() == PT.FLOWING_TEXT:
            bb = r.BoundingBox(level)
            draw.rectangle(bb, None, "orange", 8)
        elif r.BlockType() == PT.PULLOUT_TEXT:
            bb = r.BoundingBox(level)
            draw.rectangle(bb, None, "brown", 8)    
        elif r.BlockType() == PT.HEADING_TEXT:
            bb = r.BoundingBox(level)
            draw.rectangle(bb, None, "green", 8)
        elif r.BlockType() == PT.TABLE:
            bb = r.BoundingBox(level)
            draw.rectangle(bb, None, "red", 8)
        elif r.BlockType() == PT.HORZ_LINE:
            bb = r.BoundingBox(level)
            draw.rectangle(bb, None, "yellow", 8)
        elif r.BlockType() == 3:
            bb = r.BoundingBox(level)
            #print(bb)
            draw.rectangle(bb, None, "blue", 8)
#img.save(image.replace(".png", "_overlayed.jpeg"),format="jpeg")
img.show()    
