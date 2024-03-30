import cv2
import numpy as np

def empty(v):
    pass


cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)
# cap = cv2.VideoCapture(0)

cv2.createTrackbar('Blue Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Blue Max', 'TrackBar', 255, 255, empty)
cv2.createTrackbar('Green Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Green Max', 'TrackBar', 255, 255, empty)
cv2.createTrackbar('Red Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Red Max', 'TrackBar', 255, 255, empty)

img = cv2.imread('test02.jpg')
img = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    
    # ret, img = cap.read()
    # if not ret:
    #     break

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    b_min = cv2.getTrackbarPos('Blue Min', 'TrackBar')
    b_max = cv2.getTrackbarPos('Blue Max', 'TrackBar')
    g_min = cv2.getTrackbarPos('Green Min', 'TrackBar')
    g_max = cv2.getTrackbarPos('Green Max', 'TrackBar')
    r_min = cv2.getTrackbarPos('Red Min', 'TrackBar')
    r_max = cv2.getTrackbarPos('Red Max', 'TrackBar')

    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([b_min, g_min, r_min])
    upper = np.array([b_max, g_max, r_max])

    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow('frame', img)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    

    if cv2.waitKey(1) == ord('q'):
        break             # 按下 q 鍵停止

