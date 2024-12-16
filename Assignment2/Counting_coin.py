import cv2 as cv
import numpy as np

def click_event(event, x, y, flags, params,img): 
  
    # checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv.FONT_HERSHEY_SIMPLEX 
        cv.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv.imshow('image', img) 
  
    # checking for right mouse clicks      
    if event==cv.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv.imshow('image', img) 

def remove_non_unifrom_illumination(im,x):
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    background = cv.GaussianBlur(im_gray, (51, 51), 0)#cv.medianBlur(im_gray,11)
    background = np.clip(background, 1, 255)
    background_color = cv.merge([background] * 3)
    norm_im = (im / background_color) * x
    norm_im = np.clip(norm_im, 0, 255).astype(np.uint8)
    return norm_im

def counting_Bluecoin(filename):
    im =cv.imread(filename)
    target_size = (600,600)
    im = cv.resize(im,target_size)

    kernel_erodeI = np.ones((12,4),np.uint8)
    kernel_erodeL = np.ones((1,6),np.uint8)
    kernel_dilation = np.ones((20,1),np.uint8)

    norm_img = remove_non_unifrom_illumination(im,150)
    norm_img = cv.GaussianBlur(norm_img, (5, 5), 0)
    

    norm_img = cv.erode(norm_img,kernel_erodeI,iterations = 2)
    norm_img = cv.erode(norm_img,kernel_erodeL,iterations = 3)
    #norm_img = cv.dilate(norm_img,kernel_dilation,iterations = 1)

    mask_b = cv.inRange(norm_img,(165,90,0),(255,255,180))
    #mask_b = cv.dilate(mask_b,(10,1),iterations = 1)
    kernel_erode = np.ones((5,5),np.uint8)

    kernel1 = np.ones((5,5),np.uint8)

    #kernel2 = np.ones((15,15),np.uint8)
    
    mask_b = cv.morphologyEx(mask_b, cv.MORPH_OPEN, kernel1)
    #mask_b = cv.morphologyEx(mask_b, cv.MORPH_CLOSE, kernel2)
    #mask_b = cv.morphologyEx(mask_b, cv.MORPH_OPEN, kernel)
    #cv.setMouseCallback('image', click_event(norm_img)) 
    contours_blue, hierarchy_blue = cv.findContours(mask_b, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #yellow = len(contours_yellow)
    blue = len(contours_blue)
    cv.imshow("pic",im)
    cv.imshow("norm_img",norm_img)
    #cv.imshow("erode_img",erode_img)
    #cv.imshow("dilation_img",dilation_img)
    cv.imshow("mask_blue",mask_b)
    #cv.imshow("b",b)
    

    return blue

def counting_Yellowcoin(filename):
    im =cv.imread(filename)
    target_size = (600,600)
    im = cv.resize(im,target_size)

    #kernel_erodeI = np.ones((12,4),np.uint8)
    #kernel_erodeL = np.ones((1,6),np.uint8)
    #kernel_dilation = np.ones((20,1),np.uint8)

    norm_img = remove_non_unifrom_illumination(im,255)
    norm_img = cv.GaussianBlur(norm_img, (5, 5), 0)

    #norm_img = cv.erode(norm_img,kernel_erodeI,iterations = 1)
    #norm_img = cv.erode(norm_img,kernel_erodeL,iterations = 1)
    #norm_img = cv.dilate(norm_img,kernel_dilation,iterations = 1)

    mask_y = cv.inRange(norm_img,(30,240,240),(150,255,255))
    mask_y = cv.dilate(mask_y,(10,10),iterations = 1)
    kernel_erode = np.ones((5,5),np.uint8)

    kernel1 = np.ones((5,5),np.uint8)

    kernel2 = np.ones((10,10),np.uint8)
    
    mask_y = cv.morphologyEx(mask_y, cv.MORPH_OPEN, kernel1)
    #mask_y = cv.morphologyEx(mask_y, cv.MORPH_CLOSE, kernel2)
    #mask_b = cv.morphologyEx(mask_b, cv.MORPH_OPEN, kernel)
    #cv.setMouseCallback('image', click_event(norm_img)) 
    #try mode cv2.RETR_EXTERNAL and cv2.RETR_TREE
    contours_yellow, hierarchy_yellow = cv.findContours(mask_y, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #yellow = len(contours_yellow)
    yellow = len(contours_yellow)
    cv.imshow("norm_imgyellow",norm_img)
    #cv.imshow("erode_img",erode_img)
    #cv.imshow("dilation_img",dilation_img)
    cv.imshow("mask_y",mask_y)
    #cv.imshow("b",b)
    
    return yellow


anwser_blue = [8,3,4,4,7,5,3,5,6,2]
anwser_yellow = [5,6,2,2,1,3,4,5,2,4]

for i in range (1,11):
    filename = 'H:\Com vision\Assignment2\CoinCounting\coin'+str(i)+'.jpg'
    Bluecoin = counting_Bluecoin(filename)
    Yellowcoin = counting_Yellowcoin(filename)

    if Bluecoin == anwser_blue[i-1]:
        print("Blue coin Count :",Bluecoin , "ANS :", anwser_blue[i-1] ,"Yellow coin Count :",Yellowcoin , "ANS :", anwser_yellow[i-1] )
    
    cv.waitKey()