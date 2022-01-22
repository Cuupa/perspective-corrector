import io
import logging
import math
import os
import time
from sys import argv

import PIL.Image as Image
import cv2
import numpy as np
import scipy.spatial.distance
from flask import Flask, request, Response

app = Flask(__name__)

threshold_reduce_steps = 50

iso_216_ratio = 1.41


def pre_processing(img, treshold1, treshold2):
    """
    Does some pre processing to determine the document borders
    Does greyscaling, gaussian blur, cannary, dialation and erodation
    :param treshold2:
    :param treshold1:
    :param img: the image
    :return: the preprocessed image
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, treshold1, treshold2)
    kernel = np.ones((5, 5))
    img_dilation = cv2.dilate(img_canny, kernel, iterations=2)
    return cv2.erode(img_dilation, kernel, iterations=1)


def get_contours(img):
    max_area = 0
    biggest_contour = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
    reordered_points = np.zeros((4, 1, 2), np.int32)
    added = reshaped_points.sum(1)
    reordered_points[0] = reshaped_points[np.argmin(added)]
    reordered_points[3] = reshaped_points[np.argmax(added)]
    diff = np.diff(reshaped_points, 1)
    reordered_points[1] = reshaped_points[np.argmin(diff)]
    reordered_points[2] = reshaped_points[np.argmax(diff)]
    return reordered_points


def get_image(data):
    np_array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)


def process(img, treshold1, treshold2):
    preprocessed_img = pre_processing(img, treshold1, treshold2)
    biggest_contour_result = get_contours(preprocessed_img)
    if len(biggest_contour_result) == 0:
        return img
    return get_warp(img, biggest_contour_result)


def verify_image(img, extension):
    if img is None:
        return False
    return verify_color_distribution(img, extension)


def get_warp(img, biggest_contour):
    """
    Found this: https://stackoverflow.com/a/38402378
    :param biggest_contour: the biggest contour found
    :param img: the image which shall be warped
    :return:
    """
    if len(img.shape) == 1:
        return None

    (rows, cols, _) = img.shape
    # image center
    x0 = cols / 2.0
    y0 = rows / 2.0

    # image = pre_processing(img, 100, 100)
    # ordered_corners = reorder(get_contours(image))

    ordered_corners = reorder(biggest_contour)

    image_corners = [(ordered_corners[0][0][0], ordered_corners[0][0][1]),
                     (ordered_corners[1][0][0], ordered_corners[1][0][1]),
                     (ordered_corners[2][0][0], ordered_corners[2][0][1]),
                     (ordered_corners[3][0][0], ordered_corners[3][0][1])]

    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(image_corners[0], image_corners[1])
    w2 = scipy.spatial.distance.euclidean(image_corners[2], image_corners[3])

    h1 = scipy.spatial.distance.euclidean(image_corners[0], image_corners[2])
    h2 = scipy.spatial.distance.euclidean(image_corners[1], image_corners[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((image_corners[0][0], image_corners[0][1], 1)).astype('float32')
    m2 = np.array((image_corners[1][0], image_corners[1][1], 1)).astype('float32')
    m3 = np.array((image_corners[2][0], image_corners[2][1], 1)).astype('float32')
    m4 = np.array((image_corners[3][0], image_corners[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    if math.isnan(k2):
        return None

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * x0 + n23 * n33 * x0 * x0) + (
            n22 * n32 - (n22 * n33 + n23 * n32) * y0 + n23 * n33 * y0 * y0))))

    A = np.array([[f, 0, x0], [0, f, y0], [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(image_corners).astype('float32')
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

    # project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (W, H))

    _, img_result = cv2.imencode(".jpg", dst)
    return dst


def is_iso_216(height, width):
    if round(height / width, 2) == iso_216_ratio:
        return True
    # is it rotated to landscape mode?
    if round(width / height, 2) == iso_216_ratio:
        return True
    return False


def check_ratio(img):
    image = Image.open(io.BytesIO(img))
    return is_iso_216(image.height, image.width)


def verify_color_distribution(img, extension):
    """
    Verifies the image based on the occurring colors. If the number of different colors is too indistinguishable the
    image is likely to be corrupted. This may be subject to finetuning
    :param extension:
    :param img:
    :return:
    """
    _, img_result = cv2.imencode(extension, img)

    image = Image.open(io.BytesIO(img_result))
    image = image.resize((200, 200))
    cluster = {}

    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            if cluster.get(r) is not None:
                cluster[r] = cluster[r] + 1
            if cluster.get(g) is not None:
                cluster[g] = cluster[g] + 1
            if cluster.get(b) is not None:
                cluster[b] = cluster[b] + 1

            if cluster.get(r) is None:
                cluster[r] = 1
            if cluster.get(g) is None:
                cluster[g] = 1
            if cluster.get(b) is None:
                cluster[b] = 1

    # Definitely a single-colored and therefore a corrupted image.
    if len(cluster) == 1:
        logging.error("The image has only one color and is likely to be invalid")
        return False

    # If just a few colors are found, check if the color difference is noticeable
    if len(cluster) <= 5:
        keys = cluster.keys()
        if max(keys) - min(keys) <= 5:
            logging.error("The image has only " + str(
                len(cluster)) + " different colors with nearly " +
                          "indistinguishable color differences and is likely to be invalid")
            return False

    logging.error("The image seems to be valid. (different colors: " + str(len(cluster)) + ")")
    return True


@app.route("/api/image/transform", methods=['POST'])
def transform():
    start = time.perf_counter()
    files = request.files
    if not len(files):
        return Response("MISSING FILE, CHECK IF CONTENT TYPE IS MULTIPART FORM", 400)
    upload = files['key'] if 'key' in files else list(files.values())[0]

    if not upload:
        return Response("MISSING FILE", 400)

    image_name, extension = os.path.splitext(upload.filename)
    payload = upload.read()
    img = get_image(payload)

    if img is None:
        return Response("INVALID IMAGE FILE", 400)

    if check_ratio(payload):
        logging.error("This is already the correct ratio, therefore there is no need for transformation")
        return Response(payload)

    threshold1 = 150
    threshold2 = 150

    final_image = process(img, threshold1, threshold2)
    success = verify_image(final_image, extension)

    while not success:
        logging.error("Failed to get a valid image with threshold=" + str(threshold1) + " ... retrying")
        final_image = process(img, threshold1, threshold2)
        success = verify_image(final_image, extension)
        if success:
            logging.error("Got an valid image with threshold=" + str(threshold1))
        else:
            threshold1 = threshold1 - threshold_reduce_steps
            threshold2 = threshold2 - threshold_reduce_steps

        # If no valid result seems to be found, return the original image
        if threshold1 < 0 and threshold2 < 0:
            logging.error("Failed to get a valid image. Returning the original image")
            _, original_image = cv2.imencode(extension, img)
            return Response(original_image.tobytes())

    stop = time.perf_counter()
    _, img_result = cv2.imencode(extension, final_image)
    logging.error("Image transformation of '" + image_name + "' took " + str(round(stop - start, 2)) + " ms")
    return Response(img_result.tobytes())


@app.route("/api/status")
def status():
    return ""


def get_port():
    for arg in range(1, len(argv)):
        if arg == "-port":
            return argv[arg + 1]
    return 5000


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=get_port())
