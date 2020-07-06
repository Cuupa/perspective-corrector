import cv2
import numpy as np
from flask import Flask, request, Response

app = Flask(__name__)


def preProcessing(img):
    '''
    Does some pre processing to determine the document borders
    Does greyscaling, gaussian blur, cannary, dialation and erodation
    :param img: the image
    :return: the preprocessed image
    '''
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 150, 150)
    kernel = np.ones((5, 5))
    img_dialation = cv2.dilate(img_canny, kernel, iterations=2)
    return cv2.erode(img_dialation, kernel, iterations=1)


def getContours(img):
    max_area = 0
    biggest_contour = np.array([])
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            contour_perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)
            if len(approximation) == 4 and area > max_area:
                biggest_contour = approximation
                max_area = area
    return biggest_contour


def reorder(points):
    reshaped_points = points.reshape((4, 2))
    reorderd_points = np.zeros((4, 1, 2), np.int32)
    added = reshaped_points.sum(1)
    reorderd_points[0] = reshaped_points[np.argmin(added)]
    reorderd_points[3] = reshaped_points[np.argmax(added)]
    diff = np.diff(reshaped_points, 1)
    reorderd_points[1] = reshaped_points[np.argmin(diff)]
    reorderd_points[2] = reshaped_points[np.argmax(diff)]
    return reorderd_points


def getWarp(img, biggest_contour):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    point1 = np.float32(reorder(biggest_contour))
    point2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    perspective = cv2.getPerspectiveTransform(point1, point2)
    final_image = cv2.warpPerspective(img, perspective, (width, height))
    return final_image


@app.route("/api/image/transform", methods=['POST'])
def transform():
    files = request.files
    if not len(files):
        return Response("MISSING FILE, CHECK IF CONTENT TYPE IS MULTIPART FORM", 400)
    upload = files['key'] if 'key' in files else list(files.values())[0]
    if not upload:
        return Response("MISSING FILE", 400)

    data = upload.read()

    file = open(list(files.keys())[0], 'wb')
    file.write(data)
    file.close()

    np_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_array, -1)
    preprocessed_img = preProcessing(img)
    biggest_contour_result = getContours(preprocessed_img)
    final_image = getWarp(img, biggest_contour_result)
    Response.data = final_image
    return final_image


@app.route("/status", methods=['POST', 'GET'])
def status():
    return "200"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
