import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def get_best_cluster(file):

    results = pd.read_csv(file)
    results.drop(["Unnamed: 0"], axis="columns", inplace=True)

    best_ari = max(results["Adjusted Rand Index"])

    index = results.index[results["Adjusted Rand Index"] == best_ari][0]

    cols = list(results.columns)
    cols = [c for c in cols if "X" not in c]

    results.drop(cols, axis="columns", inplace=True)

    best_cluster = np.array(results.iloc[index])

    return best_cluster, best_ari


def set_axes_with_cluster(name, cluster, axes, i, base, transparency, decimals):

    path = f"solutions/{cluster}"

    files = [obj.name for obj in Path(path).iterdir()]
    result = [file for file in files if name in file][0]

    path = f"{path}/{result}"
    results, ari = get_best_cluster(path)
    results = results.reshape([64, 64])

    axes[i].imshow(base)
    axes[i].imshow(results, cmap='jet', alpha=transparency)
    axes[i].set_title(f'{cluster} ARI: {round(ari, decimals)}')
    axes[i].axis('off')
    axes[i].set_rasterized(True)


def plot_comparisson(name, decimals=3, transparency=0.5, path="solutions", fig_size=(15, 3.5)):

    image_rgb = cv2.imread(f"images/original/{name}.png")
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    image_rgb_64 = cv2.imread(f"images/resized/{name}.png")
    image_rgb_64 = cv2.cvtColor(image_rgb_64, cv2.COLOR_BGR2RGB)

    mask_image = cv2.imread(f"images/resized/{name}_mask.png")
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    algorithms = [obj.name for obj in Path(path).iterdir()]
    algorithms = [obj for obj in algorithms if "." not in obj]

    fig, ax = plt.subplots(nrows=1, ncols=len(algorithms)+2, figsize=fig_size)

    ax[0].imshow(image_rgb)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(mask_image)
    ax[1].set_title('Hand-map')
    ax[1].axis('off')

    for i, cluster in enumerate(algorithms):
        print(i+2, cluster)
        set_axes_with_cluster(name, cluster, ax, i+2, image_rgb_64, transparency, decimals)

    plt.subplots_adjust()
    title = name.replace("_", " ")
    title = title.capitalize()
    title = f"Results: {title}"
    plt.suptitle(title)

    plt.savefig(f"images/results/{name}.svg")
    plt.show()


def plot_results(images_path):

    images = [obj.name for obj in Path(images_path).iterdir()]
    images = [img.split(".")[0] for img in images]

    images = ["road_with_trees", "white_containers"]

    for i, image in enumerate(images):
        plot_comparisson(image, decimals=3, transparency=0.25, fig_size=(18, 3.5))


def main():
    plot_results(images_path="images/original")


if __name__ == '__main__':
    main()