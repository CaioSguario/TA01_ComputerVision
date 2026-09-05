from collections.abc import Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


DIRECTIONAL_ANGLES = (0.0, 22.5, 45.0, 90.0, 112.5, 135.0)


# Kernel bank with the following filters: 
#   - Directional: one for each angle in angles 
#   - circular: Gaussian and Laplacian if circular = True
class KernelBank:
    def __init__(
        self,
        angles: Sequence[float] = DIRECTIONAL_ANGLES,
        circular: bool = True,
        kernel_size: int = 9,
        gabor_sigma: float = 2.5,
        gaussian_sigma: float = 2.0,
        wavelength: float = 5.0,
        gamma: float = 0.7,
        phase: float = 0.0,
    ) -> None:

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser um inteiro positivo e ímpar.")

        if gabor_sigma <= 0 or gaussian_sigma <= 0:
            raise ValueError("Os valores de sigma devem ser positivos.")

        self.angles = tuple(angles)
        self.circular = circular
        self.kernel_size = kernel_size
        self.gabor_sigma = gabor_sigma
        self.gaussian_sigma = gaussian_sigma
        self.wavelength = wavelength
        self.gamma = gamma
        self.phase = phase
        self.kernels= self._create_kernel_bank()
        self.num_kernels=len(self.kernels)

    # Print the kernel values as floats
    def print(self) -> None:
        for name, kernel in self.kernels.items():
            print(f"\n{name}:")
            print(np.round(kernel, 3))

    # Prints the kernel as a red/blue image
    #   red coloration defines negative numbers and blue positive
    def print_as_image(self) -> None:
        number_of_kernels = len(self.kernels)
        number_of_columns = min(3, number_of_kernels)
        number_of_rows = int(np.ceil(number_of_kernels / number_of_columns))

        figure, axes = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(4 * number_of_columns, 4 * number_of_rows),
            squeeze=False,
        )

        color_map = LinearSegmentedColormap.from_list(
            "negative_red_positive_blue",
            ["red", "white", "blue"],
        )

        for axis, (name, kernel) in zip(axes.flat, self.kernels.items()):
            maximum_absolute_value = float(np.max(np.abs(kernel)))

            # TwoSlopeNorm needs limits other than zero
            if maximum_absolute_value == 0.0:
                maximum_absolute_value = 1.0

            normalization = TwoSlopeNorm(
                vmin=-maximum_absolute_value,
                vcenter=0.0,
                vmax=maximum_absolute_value,
            )

            image = axis.imshow(
                kernel,
                cmap=color_map,
                norm=normalization,
                interpolation="nearest",
            )
            axis.set_title(name.replace("_", " "))
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        # Hides empty space in the last line of kernels
        for axis in axes.flat[number_of_kernels:]:
            axis.set_visible(False)

        figure.suptitle("Banco de kernels", fontsize=14)
        figure.tight_layout()
        plt.show()
 


    def _create_kernel_bank(self) -> dict[str, np.ndarray]:

        # Directional filters
        kernels = {
            f"gabor_{angle:g}_graus": self._create_gabor_kernel(angle)
            for angle in self.angles
        }

        # Circular filters
        if self.circular:
            kernels["circular"] = self._create_circular_gaussian_kernel()
            kernels["laplaciana_circular"] = (
                self._create_laplacian_of_gaussian_kernel()
            )

        return kernels

    # Creates a square gabor kernel with size equal to kernel_size
    # Filter angulation is defined by angle_degrees
    def _create_gabor_kernel(self, angle_degrees: float) -> np.ndarray:
        theta = np.deg2rad(angle_degrees)

        kernel = cv2.getGaborKernel(
            ksize=(self.kernel_size, self.kernel_size),
            sigma=self.gabor_sigma,
            theta=theta,
            lambd=self.wavelength,
            gamma=self.gamma,
            psi=self.phase,
            ktype=cv2.CV_32F,
        )

        kernel -= kernel.mean()
        return kernel
    
    def _create_circular_gaussian_kernel(self) -> np.ndarray:
        radius_squared = self._create_radius_squared_grid()
        kernel = np.exp(-radius_squared / (2.0 * self.gaussian_sigma**2))

        kernel /= kernel.sum()

        return kernel.astype(np.float32)

    def _create_laplacian_of_gaussian_kernel(self) -> np.ndarray:
        radius_squared = self._create_radius_squared_grid()
        sigma_squared = self.gaussian_sigma**2
        gaussian = np.exp(-radius_squared / (2.0 * sigma_squared))

        kernel = ((radius_squared - 2.0 * sigma_squared) / sigma_squared**2) * gaussian

        kernel -= kernel.mean()

        return kernel

    def _create_radius_squared_grid(self) -> np.ndarray:
        radius = self.kernel_size // 2
        y, x = np.mgrid[
            -radius:radius + 1,
            -radius:radius + 1,
        ]
        return x**2 + y**2
