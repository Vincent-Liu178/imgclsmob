import numpy as np
import cv2 as cv

'''def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    weidth = image.shape[1]
    channels = image.shape[2]
    print("width: %s, height: %s, channels: %s" %(height, weidth, channels))
    for raw in range(height):
        for col in range(weidth):
            for c in range(channels):
                pv = image[raw, col, c]
                image[raw, col, c] = 255 - pv
    cv.imshow("access_pixels", image)'''

def inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("inverse demo", dst)

def creat_image():
    '''
    img = np.zeros([400, 400, 3], np.uint8)  # 0-blue,1-green,2-red so this is BGR picture.
    img[:, :, 2] = np.ones([400, 400])*255   # ':'  means keep what it was.
    cv.imshow("new_image", img)
    # we can also use [400, 400, 1] to create a single-color picture. BTW, np.uint8 is a class of array.
    '''
    m1 = np.ones([3, 3], np.uint8)
    m1.fill(127)
    print(m1)

    m2 = m1.reshape([1, 9])
    print(m2)

def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)

print("--------Hello Python !--------")
src = cv.imread(r"C:\Users\sprin\Desktop\practicePy\img\Hog.jpg")
cv.namedWindow("input image", cv.WINDOW_NORMAL)
cv.imshow("input image", src)
t1 = cv.getTickCount()
#access_pixels(src)
t2 = cv.getTickCount()
color_space_demo(src)
time = 1000*(t2 - t1)/cv.getTickFrequency()
print("time: %s ms" %time)

cv.waitKey(0)
cv.destroyAllWindows()