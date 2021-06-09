



























                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            cv2.line(crop_image, start, end, [0, 255, 0], 2)
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(frame, "ONE ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(frame, "THREE ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(frame, "TWO ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            if angle <= 90:
            pass
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
        # Find convexity defects
        # print(defects.shape)
        # use cosine rule to find angle
        break
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        count_defects = 0
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
        defects = cv2.convexityDefects(contour, hull)
        drawing = np.zeros(crop_image.shape, np.uint8)
        elif count_defects == 4:
        else:
        for i in range(defects.shape[0]):
        hull = cv2.convexHull(contour)
        hull = cv2.convexHull(contour, returnPoints=False)
        if count_defects == 0:
        if count_defects == 1:
        if count_defects == 2:
        if count_defects == 3:
        pass
        x, y, w, h = cv2.boundingRect(contour)
    all_image = np.hstack((drawing, crop_image))
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    crop_image = frame[100:300, 100:300]
    cv2.imshow("Countours", all_image)
    cv2.imshow("Gesture", frame)
    cv2.imshow("Threshold", thresh)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 0)
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    except:
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    if cv2.waitKey(1) == ord('q'):
    kernel = np.ones((5, 5))
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    ret, frame = capture.read()
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    try:
# Developed by Amresh Ranjan.
capture = cv2.VideoCapture(0)
capture.release()
cv2.destroyAllWindows()
import cv2
import math
import numpy as np
while capture.isOpened():