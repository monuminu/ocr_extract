import cv2
import numpy as np
import pandas as pd
import pytesseract

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou, interArea

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []    
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def convert_img_csv(img_file_name, csv_file_name):

    img = cv2.imread(img_file_name)
    squares = find_squares(img)

    squares_mod = []
    for square in squares:
        x1,y1,x2,y2 = square[0][0],square[0][1],square[2][0],square[2][1]
        squares_mod.append([x1,y1,x2,y2])
        
    uniqueList=[]
    for letter in squares_mod:
        if letter not in uniqueList:
            uniqueList.append(letter)   

    to_be_removed_index = []
    for i,square in enumerate(uniqueList):
        count = 0
        for j,inner_square in enumerate(uniqueList):
            inu, ins = bb_intersection_over_union(square,inner_square)
            if  ins > 0:
                square_area = (square[2] - square[0] + 1) * (square[3] - square[1] + 1)
                inner_square_are = (inner_square[2] - inner_square[0] + 1) * (inner_square[3] - inner_square[1] + 1)
                if square_area > inner_square_are:
                    count = count + 1
        if count > 0:
            to_be_removed_index.append(i)

    r = [i for j, i in enumerate(uniqueList) if j not in to_be_removed_index]

    for item in r:
        x1,y1,x2,y2 = item                           
        roi = img[y1:y2,x1:x2]                        

    df = pd.DataFrame(r)
    e = df.sort_values([1,0],axis = 0)

    config = ('-l eng --oem 3 --psm 3')
    def detect_text_ocr(image):
        text = pytesseract.image_to_string(image, config=config)
        return text

    e['text'] = ''
    for index, row in e.iterrows():
        x1,y1,x2,y2 = row[0], row[1], row[2], row[3]
        roi = img[y1:y2,x1:x2]
        txt = detect_text_ocr(roi)
        e.loc[index,'text'] = txt

    prev_col = 4
    ths = 5
    file = None
    for index, row in e.iterrows():
        x1,y1,x2,y2 = row[0], row[1], row[2], row[3]
        if y1 - prev_col < ths:
            if file is None:
                file = str(row[4])
            else:
                file = file + ',' +str(row[4])
        else:
            file = file + '\n' + str(row[4])
            prev_col = y1

    f = open(csv_file_name, 'w' )
    f.write(file)
    f.close()

if __name__ = '__main__':
    filename = 'a.jpg'
    src_img = cv2.imread(filename)

    if not src_img.any():
        print('Problem loading data')
        exit

    resized_img = src_img.copy()
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    black_white_img = cv2.adaptiveThreshold(cv2.bitwise_not(gray_img), 255, 
                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, 15, -2)

    horizontal = black_white_img.copy()
    vertical   = black_white_img.copy()

    scale = 30
    rows,cols = horizontal.shape
    horizontalsize = int(cols / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

    scale = 15
    verticalsize = int(rows / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, ( 1,verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    mask = horizontal + vertical

    _, contours, hierarchy= cv2.findContours(mask, 
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    #biggest rectangle
    size_rectangle_max = 0
    boundRect = [0] * len(contours) 
    contours_poly = [0] * len(contours)
    rois = []

    for i in range(len(contours)):
        print(i)
        area = cv2.contourArea(contours[i])
        if area < 100:
            continue
        #aproximate countours to polygons
        contours_poly[i] = cv2.approxPolyDP(contours[i], 4, True)
        boundRect[i] = cv2.boundingRect( (contours_poly[i]) )
        x, y, width, height = boundRect[i]
        roi = black_white_img[y:y+height, x:x+width]
        _,joints_contours,_ = cv2.findContours(roi, 
                                cv2.RETR_CCOMP, 
                                cv2.CHAIN_APPROX_SIMPLE)
        #has the polygon 4 sides?
        #if(not (len(joints_contours)==4)):
        #    continue
        print(len(joints_contours))
        if len(joints_contours) < 50:
            continue
        rois.append(src_img[y:y+height, x:x+width].copy())
        show_table = False
        if show_table:  
            cv2.rectangle(src_img, (x,y),(x+width,y+height),(0, 255, 0),2)
            cv2.namedWindow('img1', 0)
            cv2.imshow('img1', resized_img)
            while(cv2.waitKey()!=ord('q')):
                continue
            cv2.destroyAllWindows()

    for i, roi in enumerate(rois):
        img_file_name = str(i) + '_table.jpg'
        csv_file_name = str(i) + '_table.csv'
        cv2.imwrite(img_file_name,roi)
        convert_img_csv(img_file_name, csv_file_name)

