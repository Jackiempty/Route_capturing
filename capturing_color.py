import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
# image = mpimg.imread('test02.jpg')
# cap = cv2.VideoCapture('./test_videos/test02.mp4')
cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
# cap = cv2.VideoCapture('./test_videos/solidYellowLeft.mp4')
# cap = cv2.VideoCapture('./test_videos/challenge.mp4')
# image = cv2.imread('test.jpg')

while True:
    
    ret, image = cap.read()
    if not ret:
        break


    image = cv2.resize(image, (0, 0), fx = 0.8, fy = 0.8)
    # Grab the x and y sizes and make two copies of the image
    # With one copy we'll extract only the pixels that meet our selection,
    # then we'll paint those pixels red in the original image to see our selection 
    # overlaid on the original.
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select= np.copy(image)
    line_image = np.copy(image)

    # Define our color criteria
    blue_threshold = 0
    green_threshold = 150
    red_threshold = 210
    rgb_threshold = [blue_threshold, green_threshold, red_threshold]

    # Define a triangle region of interest (Note: if you run this code, 
    # Keep in mind the origin (x=0, y=0) is in the upper left in image processing
    # you'll find these are not sensible values!!
    # But you'll get a chance to play with them soon in a quiz ;)
    left_bottom = [xsize*150/1915, ysize*1000/985]
    right_bottom = [xsize*1900/1915, ysize*1000/985]
    apex = [xsize*950/1915, ysize*550/985]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                        (image[:,:,1] < rgb_threshold[1]) | \
                        (image[:,:,2] < rgb_threshold[2])

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Mask color selection
    color_select[color_thresholds] = [0,0,0]
    # Find where image is both colored right and in the region
    line_image[~color_thresholds & region_thresholds] = [0,0,255]

    # cvt_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

    # Display our two output images
    cv2.imshow("line_image", line_image)

    if cv2.waitKey(1) == ord('q'):
        break             # 按下 q 鍵停止
