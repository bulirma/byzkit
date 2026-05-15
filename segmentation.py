import cv2
import numpy as np


def neume_x_bounds(img: cv2.Mat):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def get_x(contour):
        x, _, w, _ = cv2.boundingRect(contour)
        return x + w / 2

    contours = sorted(contours, key=get_x)

    neumes = []
    current_neume = [contours[0]]
    
    for c in contours[1:]:
        x_prev, _, w_prev, _ = cv2.boundingRect(current_neume[-1])
        x, _, _, _ = cv2.boundingRect(c)
        if x - (x_prev + w_prev) < 6:
            current_neume.append(c)
        else:
            neumes.append(current_neume)
            current_neume = [c]
    neumes.append(current_neume)

    bounds = []
    for i, neume in enumerate(neumes):
        points = np.vstack(neume).reshape(-1, 2)
        lbound = np.min(points[:, 0]) - 1
        rbound = np.max(points[:, 0]) + 1
        bounds.append((lbound, rbound))

    return bounds

def get_line_bboxes(img: cv2.Mat) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    closing_kernel = np.ones((12, 88), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)
    dilatation_kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(closed, dilatation_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if h > 30 and w > 150 and area > 1000:
            line_contours.append((c, y))
    
    line_contours.sort(key=lambda x: x[1])

    line_bboxes = []
    for i, (contour, _) in enumerate(line_contours):
        x, y, w, h = cv2.boundingRect(contour)
        l = x - 2
        r = x + w + 2
        t = y - 2
        b = y + h + 2
        line_bboxes.append((t, b, l, r))

    return line_bboxes

def get_line_images(img: cv2.Mat):
    bboxes = get_line_bboxes(img)
    line_imgs = []
    for t, b, l, r in bboxes:
        line_img = img[t: b, l: r]
        line_imgs.append(line_img)
    return line_imgs

def get_color_bbox(img: cv2.Mat, color: tuple) -> tuple:
    mask = (img == color).all(axis=2)
    coords = np.column_stack(np.nonzero(mask))
    ys = coords[:, 0]
    xs = coords[:, 1]
    t = min(ys)
    b = max(ys)
    l = min(xs)
    r = max(xs)
    return (t, b, l, r)

def get_neume_images(line_img: cv2.Mat) -> list:
    imgs = []
    neume_bounds = neume_x_bounds(line_img)
    for l, r in neume_bounds:
        neume_img = line_img[:, l - 1: r + 2]
        imgs.append(neume_img)
    return imgs

def count_neume_images(line_img: cv2.Mat) -> int:
    return len(neume_x_bounds(line_img))

def get_line_images_with_neume_count(img: cv2.Mat) -> (list, list):
    line_imgs = get_line_images(img)
    neume_counts = [count_neume_images(line) for line in line_imgs]
    return line_imgs, neume_counts
