import cv2
import matplotlib.pyplot as plt
import numpy as np



def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000 and area<10000000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def warping(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_original=img.copy()
    img_c=img.copy()
    blur=cv2.GaussianBlur(gray,(5,5),0)
    #res1, th1 = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    #res1,th1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th1 = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    edged=cv2.Canny(th1,100,200)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    dilation=cv2.dilate(edged,kernel,iterations=2)
    
##    plt.imshow(th1)
##    plt.show()
    #contours, hierachy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)
    cv2.drawContours(img_c, [biggest], -1, (0, 255, 0), 3)
    plt.imshow(img_c, 'Greys')
    plt.axis('off')
    plt.show()
    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    # max_height = max(int(right_height), int(left_height))
    max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    resized_image = cv2.resize(img_output,dsize=(2205,3117))
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()
##    print(resized_image.shape)
    return resized_image

def templateMatching(img,temp1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(img_gray,(5,5),0)
    #res2, th2 = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    #res2,th2=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #res3, th3 = cv2.threshold(temp1,100,255,cv2.THRESH_BINARY)
    
##    plt.imshow(temp1)
##    plt.show()
    h, w = temp1.shape

    method = cv2.TM_CCOEFF
    img2 = th2.copy()

    #matching template 1
    result = cv2.matchTemplate(th2,temp1,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location =  max_loc
    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 0, 3)

##    #matching template 2
##    result2 = cv2.matchTemplate(th2,th4,method)
##    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)
##    location2 =  max_loc
##    bottom_right2 = (location2[0] + w2, location2[1] + h2)
##    cv2.rectangle(img2, location2, bottom_right2, 0, 3)

##    t1=(bottom_right2[0]+30,bottom_right[1]+50)
##    b1=(bottom_right2[0]+300,bottom_right[1]+155)

    if location[0]<1000:
        t1=(location[0]+180,bottom_right[1]+50)
        b1=(location[0]+450,bottom_right[1]+154)
    else:
        t1=(location[0]-630,bottom_right[1]+50)
        b1=(location[0]-370,bottom_right[1]+154)
    
    plt.imshow(img2,'Greys_r')
    plt.show()
    return img2,t1,b1,location[0]

def comparison(img1,img2,t1,b1,br1,t2,b2,br2):
    t11=t1
    b11=b1
    br11=br1
    t22=t2
    b22=b2
    br22=br2
    count=[]
    diff=[]
    for j in range(20):
        for i in range(5):
            cv2.rectangle(img1,t1,b1,0,3)
            cv2.rectangle(img2,t2,b2,0,3)
            check1=img1[t1[1]:b1[1],t1[0]:b1[0]]
            check2=img2[t2[1]:b2[1],t2[0]:b2[0]]
##            plt.imshow(check1)
##            plt.show()
##            plt.imshow(check2)
##            plt.show()
            count1 = cv2.countNonZero(check1)
            count2 = cv2.countNonZero(check2)
            diff.append(count1-count2)
            t1=(t1[0]+260,t1[1])
            b1=(b1[0]+260,b1[1])
            t2=(t2[0]+260,t2[1])
            b2=(b2[0]+260,b2[1])
        if br2<1000:
            t2=(br2+180,t2[1]+104)
            b2=(br2+450,b2[1]+104)
        else:
            t2=(br2-630,t2[1]+104)
            b2=(br2-370,b2[1]+104)
        t1=(br1-630,t1[1]+105)
        b1=(br1-370,b1[1]+105)

    #print(diff)
    maximum= max(diff)
    average= np.mean(diff)
##    print('max: ',maximum)
##    print('mean: ',average)
    difference=[]
    for j in range(20):
        for i in range(5):
            cv2.rectangle(img1,t11,b11,0,3)
            cv2.rectangle(img2,t22,b22,0,3)
            check1=img1[t11[1]:b11[1],t11[0]:b11[0]]
            check2=img2[t22[1]:b22[1],t22[0]:b22[0]]
##            plt.imshow(check1)
##            plt.show()
##            plt.imshow(check2)
##            plt.show()
            count1 = cv2.countNonZero(check1)
            count2 = cv2.countNonZero(check2)
            #print(count1-count2)
            difference.append(count1-count2)
            if count1-count2>average+200:#maximum-505:
                count.append(1)
            else:
                count.append(0)
            
            t11=(t11[0]+260,t11[1])
            b11=(b11[0]+260,b11[1])
            t22=(t22[0]+260,t22[1])
            b22=(b22[0]+260,b22[1])
        if br22<1000:
            t22=(br22+180,t22[1]+104)
            b22=(br22+450,b22[1]+104)
        else:
            t22=(br22-630,t22[1]+104)
            b22=(br22-370,b22[1]+104)
        t11=(br11-630,t11[1]+105)
        b11=(br11-370,b11[1]+105)    
    arr=np.array(count)
##    print(arr.reshape(20,5))
##    print(len(difference))
    responses=arr.reshape(20,5)
    arr2=np.array(difference)
    differences=arr2.reshape(20,5)
    plt.imshow(img1,'Greys_r')
    plt.show()
    plt.imshow(img2,'Greys_r')
    plt.show()
    print(responses)
    return responses,differences

def exceptions(count,difference):
##    print(difference)
    for i in range(20):
        arr=[]
        arr1=[]
        for j in range(5):
            arr1.append(difference[i,j])
            if count[i,j]==1:
                arr.append(j)
        j=0
        print(arr)
        if len(arr)>1:
            arr2=[]
            arr2.append(difference[i,int(arr[0])])
            arr2.append(difference[i,int(arr[1])])
            
            #arr2=np.array(arr2)
            arr2.sort()
            print(arr2)
            if arr2[1]-arr2[0]>=500:
                position=arr1.index(arr2[0])
                for a in range(5):
                    if a==position:
                        count[i,a]=1
                    else:
                        count[i,a]=0
                a=0
            else:
                for a in range(5):
                    count[i,a]=0
                a=0
        else:
            continue

    print(count)
    return count

def calc_marks(a_answers, m_answers):
    count=0
    for i in range(20):
        if a_answers[i,0]==m_answers[i,0]and a_answers[i,1]==m_answers[i,1]and a_answers[i,2]==m_answers[i,2]and a_answers[i,3]==m_answers[i,3]and a_answers[i,4]==m_answers[i,4]:
            count=count+1
        else:
            continue
    return count
                
            


img_unmarked=cv2.imread(r'C:/Users/Nilni Kamburugamuwa/Downloads/test3.jpg')
img_marked=cv2.imread(r'C:/Users/Nilni Kamburugamuwa/Downloads/test35.jpg')
img_output1=warping(img_unmarked)
img_marked_thresh=cv2.bilateralFilter(img_marked,5,75,75)
img_output2=warping(img_marked)

template1 = cv2.resize(cv2.imread('C:/Users/Nilni Kamburugamuwa/Downloads/template_cropped.jpg',0), (0,0), fx=0.4, fy=0.4)
img1,t1,b1,br1=templateMatching(img_output1,template1)
img2,t2,b2,br2=templateMatching(img_output2,template1)
responses,differences=comparison(img1,img2,t1,b1,br1,t2,b2,br2)
marked_answers=exceptions(responses,differences)
actual_answers=np.array([[0,0,1,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [1,0,0,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
                [0,0,0,0,1],
                [0,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,0,0,0,0],
                [0,1,0,0,0]])
marks=calc_marks(actual_answers,marked_answers)
print('Total marks: ',marks)
