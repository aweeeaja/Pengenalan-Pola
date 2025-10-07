import cv2
import numpy as np
import os

path = 'image'

sift = cv2.SIFT_create()

images = []
className = []
myList = os.listdir(path)
print('total class',len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    if imgCur is not None:
        images.append(imgCur)
        className.append(os.path.splitext(cl)[0])
    else:
        print(f'Error reading file {cl}')
print(className)

def findDes(images):
    deslist= []
    for img in images:
        kp,des = sift.detectAndCompute(img, None)
        if kp is not None and des is not None:
            deslist.append(des)
        else:
            print('Error computing descriptors')
    return deslist

def findId(img, deslist, thres=10):
    kp2,des2 = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in deslist:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
        if len(matchList) != 0:
            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))
    except:
        pass
    return finalVal

deslist = findDes(images)
print(deslist)

cap = cv2.VideoCapture(1)

threshold = 10
while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    id = findId(img2, deslist, threshold)
    if id !=-1:
        cv2.putText(imgOriginal,className[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv2.imshow('img2',imgOriginal)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()