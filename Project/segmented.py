import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from skimage.filters import threshold_multiotsu


def segment_image(image_number, save_path="outputs"):

    # Setting the font size for all plots.
    matplotlib.rcParams['font.size'] = 9

    # The input image.
    image = cv2.imread(f'image/{image_number}mapa.png', 0)
    image_rgb = cv2.imread(f'image/{image_number}.png')
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(image)
    ax[1].set_title('Hand-map')
    ax[1].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    #ax[2].hist(image.ravel(), bins=255)
    #ax[2].set_title('Histogram')
    #for thresh in thresholds:
    #    ax[2].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    ax[3].imshow(image_rgb)
    ax[3].imshow(regions, cmap='jet', alpha=0.3)
    ax[3].set_title('Overlap')
    ax[3].axis('off')



    plt.subplots_adjust()
    plt.savefig(f"outputs/otsu/{image_number}_otsu_class.png")
    plt.show()

    path_x = f"{save_path}/{image_number}_x.csv"
    path_y = f"{save_path}/{image_number}_y.csv"

    if not (Path(path_x).is_file() and Path(path_y).is_file()):

        file_x = open(path_x, "a")
        file_y = open(path_y, "a")

        for x in range(len(regions)):
            for y in range(len(regions[0])):
                r, g, b = image_rgb[x][y]
                output = f'{r},{g},{b}'
                label = str(regions[x][y])
                file_x.write(f"{output}\n")
                file_y.write(f"{label}\n")
    else:
        print("file already exists")


def main():
    for i in range(1, 8, 1):
        segment_image(image_number=i)


if __name__ == '__main__':
    main()
