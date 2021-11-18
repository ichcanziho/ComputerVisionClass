import cv2
import numpy as np
from random import choice, randint
import pandas as pd


def get_new_images(image, contours, item_num):
    mask = np.zeros(image.shape, np.uint8)
    new_image = cv2.drawContours(mask, contours,  item_num, (255,255,255), 1)
    return new_image


def segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    objects = {}
    for item in range(len(contours)):
        objects[item] = get_new_images(image, contours, item)
    return objects, contours


def get_features(image):
    objects, contours = segmentation(image)
    features = []
    for item in range(len(contours)):
        M = cv2.moments(contours[item])
        hu_moments = cv2.HuMoments(M)
        features.append(hu_moments[0][0])
        features.append(hu_moments[1][0])
        features.append(hu_moments[2][0])
        features.append(hu_moments[3][0])
        features.append(hu_moments[4][0])
        features.append(hu_moments[5][0])
        features.append(hu_moments[6][0])
        area = cv2.contourArea(contours[item])
        features.append(area)
        perimeter = cv2.arcLength(contours[item], True)
        features.append(perimeter)
        try:
            features.append(4 * np.pi * area / (perimeter**2))
        except ZeroDivisionError:
            features.append(0)
    return features


def create_char_image(char, angle, font, scale, thickness):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    text_location = (randint(50, 150), randint(50, 150))
    cv2.putText(img, char, text_location, font, scale, (255, 255, 255), thickness)
    M = cv2.getRotationMatrix2D(text_location, angle, 1)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return cv2.bitwise_not(out)


def create_char_images(chars, n, output_dir="database"):
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC]

    entries = []
    for char in chars:
        print("current:", char)
        i = 1
        while i <= 100:
            image = create_char_image(char=char, angle=choice([1, -1])*randint(0, 30),
                                      font=choice(fonts), scale=randint(1, 3), thickness=randint(1, 3))
            features = get_features(image)
            features.append(char)
            if len(features) == 11:
                entries.append(features)
                cv2.imwrite(f"images/{char}/{i}.png", image)
                print(f"iteration: {i}/{n}")
                i += 1
        print("")
    cols = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "area", "perimeter", "compactness", "class"]

    frame = pd.DataFrame(data=entries, columns=cols)
    frame.to_csv(f"{output_dir}/database.csv", index=False)


def main():
    chars = ["a", "b", "c"]
    create_char_images(chars, 100)


if __name__ == '__main__':
    main()
