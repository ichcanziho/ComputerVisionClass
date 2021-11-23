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

    results.drop(["Dataset", "Algorithm", "Clusters", "Instances", "Features",
                  "Pop. size", "Max. gens", "Gens", "No. objectives",
                  "Obj. name", "Fitness", "Time", "Adjusted Rand Index"], axis="columns", inplace=True)

    best_cluster = np.array(results.iloc[index])

    return best_cluster, best_ari


def plot_comparisson(name, decimals=3, transparency=0.5, ecac_path="solutions/F1ECAC", kmeans_path="solutions/kMeans"):

    F1ECAC_Files = [obj.name for obj in Path(ecac_path).iterdir()]
    F1ECAC_Result = [file for file in F1ECAC_Files if name in file][0]

    kMeans_Files = [obj.name for obj in Path(kmeans_path).iterdir()]
    kMeans_Result = [file for file in kMeans_Files if name in file][0]

    ecac_path = f"{ecac_path}/{F1ECAC_Result}"
    kmeans_path = f"{kmeans_path}/{kMeans_Result}"

    ecac_cluster, ecac_ari = get_best_cluster(ecac_path)
    ecac_cluster = ecac_cluster.reshape([64, 64])

    kmans_cluster, kmeans_ari = get_best_cluster(kmeans_path)
    kmans_cluster = kmans_cluster.reshape([64, 64])

    image_rgb = cv2.imread(f"images/original/{name}.png")
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    image_rgb_64 = cv2.imread(f"images/resized/{name}.png")
    image_rgb_64 = cv2.cvtColor(image_rgb_64, cv2.COLOR_BGR2RGB)

    mask_image = cv2.imread(f"images/resized/{name}_mask.png")
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3.5))

    ax[0].imshow(image_rgb)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(mask_image)
    ax[1].set_title('Hand-map')
    ax[1].axis('off')

    ax[2].imshow(image_rgb_64)
    ax[2].imshow(ecac_cluster, cmap='jet', alpha=transparency)
    ax[2].set_title(f'F1-ECAC ARI: {round(ecac_ari, decimals)}')
    ax[2].axis('off')
    ax[2].set_rasterized(True)

    ax[3].imshow(image_rgb_64)
    ax[3].imshow(kmans_cluster, cmap='jet', alpha=transparency)
    ax[3].set_title(f'k-Means ARI: {round(kmeans_ari, decimals)}')
    ax[3].axis('off')
    ax[3].set_rasterized(True)

    plt.subplots_adjust()
    title = name.replace("_", " ")
    title = title.capitalize()
    title = f"Results: {title}"
    plt.suptitle(title)

    # Retrieve a view on the renderer buffer

    plt.savefig(f"images/results/{name}.svg")


def plot_results(images_path):

    images = [obj.name for obj in Path(images_path).iterdir()]
    images = [img.split(".")[0] for img in images]

    for i, image in enumerate(images):
        plot_comparisson(image, decimals=3, transparency=0.25)


def main():
    plot_results(images_path="images/original")


if __name__ == '__main__':
    main()