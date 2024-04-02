# Finding Lane Lines on the Road 

There are couple of ways to implement route capturing, and the way I chose to try is is to implement it with color selection.  

For you to know, the color composition of an image is both the color and the brightness. Take a JPEG image as an instance, it is composed of three layers of color which is Red, Green, and Blue respectively. For each color has its own transparency, and an image is the combination of all three color with it transparency in each pixel, which build up the whole picture with lots of pixels.  

![image](https://hackmd.io/_uploads/HkhDzR8y0.png)  
> source: https://en.wikipedia.org/wiki/Pixel_art  

So how exactly to recognize the lane shape just by detecting color?   

## Find color

The first step is to find the specific color of the marking on the road. To approach that, I create an simple tool with opencv to specify the color of the marking.

**before:**
![image](https://hackmd.io/_uploads/ryZJSAIJA.png)


**after:**
![image](https://hackmd.io/_uploads/B1ak408yC.png)

As you can see, by dragging the adjustment bar, it becomes possible to specify the color just for the marking on the road, so that we can capture the route.

```python
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

img = cv2.imread('./test_images/challange02.jpg')
img = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:

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
```
> This is the program used to specify the demanding color

## Region Mask

After finding the wanted color, there's another problem which is the surrounding of the road that may be noisy when we want only the color of the marking on the road to be specify rather that the whole image including the sky or other cars.  

So what we do is to make a region mask to cover only the region that we want to process.

![image](https://hackmd.io/_uploads/SJCyuCIkC.png)  

We approach it by using the plotting tool privided by the package of the assignment.

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('test_images/challange.jpg')
print('This image is: ', type(image),'with dimensions:', image.shape)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz 
left_bottom = [xsize*200/1915, ysize*1000/985]
right_bottom = [xsize*1800/1915, ysize*1000/985]
apex = [xsize*950/1915, ysize*550/985]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

# Display the image
#check matplotlib.pyplot.plot (https://matplotlib.org/api/pyplot_api.html#module-matplotlib.pyplot)
#tip:plt.plot([0], [539], 'bo')
plt.imshow(region_select)
```
By providing three coordinate of the vertex of the triangle, we get to draw a shape just to cover the region we want to process.  



## Result

By combining the two processes together, we get to aquire a program that can specify the color of the marking on the road just inside the region where the marking exists. 

![image](https://hackmd.io/_uploads/ByAW90IJ0.png)


**code**
```python
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
cap = cv2.VideoCapture('./test_videos/challenge.mp4')

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

```
# Link
[solidWhiteRight](https://youtu.be/fEzKZc-sFc4)
[solidYellowLeft](https://youtu.be/xJ65LmKLL3A)
[challenge](https://youtu.be/YrKArkoXF10)
[bonus](https://youtu.be/-QsYOkL1XoM)

https://youtu.be/dyHaMPcWi0g
# Things to improve
- [ ] connect all the line together to form a bigger and more visable line  
- [ ] intruduce Gauss blur, canny edge and hough transform to enhance the robustness of the program