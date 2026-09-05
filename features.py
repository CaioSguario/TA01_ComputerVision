import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from image import Image
from kernel_bank import KernelBank

class Features:
    def __init__(
                self,
                kernel_bank:KernelBank = None,
                image:Image = None,
                patches_x:int = 32,
                patches_y:int = 32,
                scales:int = 3,
                k_clusters:int = 6,
            ) -> None:

            if kernel_bank is None or image is None:
                raise ValueError("kernel_bank and image can't be None")

            if scales < 1:
                raise ValueError("scales must be an integer equal or bigger than one")

            if patches_x  < 1 or patches_y < 1:
                raise ValueError("patches_x and patches_y must be an integer equal or bigger than one")

            if k_clusters < 1:
                raise ValueError("k_clusters must be an integer equal or bigger than one")

            if k_clusters > patches_x * patches_y:
                raise ValueError("k_clisters can't be bigger than the number of patches")

            self.kernel_bank = kernel_bank
            self.image = image
            self.patches_x = patches_x
            self.patches_y = patches_y
            self.scales = scales

            
            self.feature_names = tuple(
                f"scale_{scale}_{kernel_name}"
                for scale in range(self.scales)
                for kernel_name in tuple(kernel_bank.kernels.keys())
            )


            # Matrix of features for each patch of the image
            # Each image will be divided into patches_x * patches_y patches
            self.feature_size = len(self.feature_names)
            self.features = np.zeros((patches_y, patches_x, self.feature_size), dtype=np.float32)


            # Does the textural feature extraction
            self._extract_features()
            
            # Cluster creation on gathered features 
            self.cluster_labels:np.ndarray | None = None
            self.kmeans:KMeans | None = None
            self.k_clusters = k_clusters
            self._cluster_patches()

    def show_features_as_image(self) -> None:
        if self.features.size == 0:
            raise ValueError("No features were extracted.")

        num_kernels = len(self.kernel_bank.kernels)

        figure, axes = plt.subplots(
            nrows=self.scales,
            ncols=num_kernels,
            figsize=(4 * num_kernels, 4 * self.scales),
            squeeze=False,
        )

        for scale in range(self.scales):
            for kernel_index, kernel_name in enumerate(
                self.kernel_bank.kernels.keys()
            ):
                feature_index = (
                    scale * num_kernels
                    + kernel_index
                )

                # Matriz patches_y × patches_x da feature atual.
                feature_map = self.features[
                    :,
                    :,
                    feature_index,
                ]

                axis = axes[scale, kernel_index]

                image = axis.imshow(
                    feature_map,
                    cmap="viridis",
                    interpolation="nearest",
                    origin="upper",
                )

                axis.set_title(
                    f"Escala {scale + 1}\n"
                    f"{kernel_name.replace('_', ' ')}"
                )

                axis.set_xlabel("Patch X")
                axis.set_ylabel("Patch Y")

                axis.set_xticks(range(self.patches_x))
                axis.set_yticks(range(self.patches_y))

                figure.colorbar(
                    image,
                    ax=axis,
                    fraction=0.046,
                    pad=0.04,
                )

        figure.suptitle(
            "Características de textura por patch",
            fontsize=16,
        )

        figure.tight_layout()
        plt.show()

    # Shows the original image with colored clusters ontop, showcasing the cluster segmentation
    def show_segmented_image(self) -> None:
        # Creates the cluster segmentation map
        segmentation = self._create_segmentation_mask()

        # Showcases the cluster segmentation map in color and overlapped with original image
        image = self.image.image

        plt.figure()

        if len(image.shape) == 2:
            plt.imshow(
                    image,
                    cmap="gray",
                )
        else:
            plt.imshow(image)

        plt.imshow(
                segmentation,
                cmap="tab20",
                alpha=0.4,
                interpolation="nearest",
            )

        plt.title("Texture Segmentation")
        plt.axis("off")
        plt.show()


    def _create_segmentation_mask(self) -> np.ndarray:
        image = self.image.image

        height,width = image.shape[:2]

        mask = np.zeros((height, width), dtype=np.int32)

        # Patch limits
        x_edges = np.linspace(0, width, self.patches_x + 1, dtype=int)
        y_edges = np.linspace(0, height, self.patches_y + 1, dtype=int)

        for patch_y in range(self.patches_y):
            for patch_x in range(self.patches_x):
                cluster = self.cluster_labels[patch_y, patch_x]

                x_start = x_edges[patch_x]
                x_end = x_edges[patch_x + 1]

                y_start = y_edges[patch_y]
                y_end = y_edges[patch_y + 1]

                # All pixels in the current patch are inside its cluster
                mask[y_start:y_end, x_start:x_end] = cluster

        return mask 


    def _cluster_patches(self, random_state:int = 42) -> None:
        # KMeans needs a matrix of dimension 2
        features_flat = self.features.reshape(-1, self.feature_size)

        # Normalization of the values
        scaler = StandardScaler()
        features_flat = scaler.fit_transform(features_flat)
        self.feature_scaler = scaler

        kmeans = KMeans(
                n_clusters=self.k_clusters,
                random_state=99,
                n_init="auto",
            )

        labels = kmeans.fit_predict(features_flat)

        # Returns to a matrix of dimension 3
        self.cluster_labels = labels.reshape(
                self.patches_y,
                self.patches_x,
            )

        self.kmeans = kmeans
        

    def _extract_features(self) -> None:
        scale_image = self.image

        for scale in range(self.scales):
            height = scale_image.height
            width = scale_image.width

            # Verifies if image is big enough to divide into the specified patches
            if height < self.patches_y or width < self.patches_x:
                raise ValueError(f"Image at scale {scale} ({width}x{height}) is smaller than the patch grid")
            
            y_limits = np.linspace(0, height, self.patches_y + 1, dtype=int)
            x_limits = np.linspace(0, width, self.patches_x + 1, dtype=int)

            for kernel_index, kernel_name in enumerate(tuple(self.kernel_bank.kernels.keys())):
                # Copies the current kernel to avoid wrong modification of the original
                kernel = np.asarray(self.kernel_bank.kernels[kernel_name], dtype=np.float32)

                # Passes the kernel in the whole image, and latter calculates patch value
                # This approach aims to mitigate patch border loss of information
                response = cv2.filter2D(
                                        scale_image.image,
                                        ddepth=cv2.CV_32F,
                                        kernel=kernel,
                                        borderType=cv2.BORDER_REFLECT_101,
                            )

                absolute_response = np.abs(response)
                current_feature_index = scale * self.kernel_bank.num_kernels + kernel_index

                for patch_y in range(self.patches_y):
                    y_start, y_end = y_limits[patch_y : patch_y + 2]
                    for patch_x in range(self.patches_x):
                        x_start, x_end = x_limits[patch_x : patch_x + 2]
                        patch_total_value = absolute_response[y_start:y_end, x_start:x_end]
                        self.features[patch_y, patch_x, current_feature_index] = patch_total_value.mean()

            if scale + 1 < self.scales:
                scale_image = scale_image.gaussian_downscale()
    
