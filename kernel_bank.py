from collections.abc import Sequence

import cv2
import numpy as np


DIRECTIONAL_ANGLES = (0.0, 22.5, 45.0, 90.0, 112.5, 135.0)


# Banco de kernels contendo:
# Filtros direcionais -> angulos definidos por angles
# Filtros circulares (gaussiano e laplaciano do gaussiano) -> circular = True
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

    def print(self) -> None:
        for name, kernel in self.kernels.items():
            print(f"\n{name}:")
            print(np.round(kernel, 3))


    # Cria definitivamente o banco do kernel
    def _create_kernel_bank(self) -> dict[str, np.ndarray]:

        # Filtros direcionais
        kernels = {
            f"gabor_{angle:g}_graus": self._create_gabor_kernel(angle)
            for angle in self.angles
        }

        # Filtros circulares
        if self.circular:
            kernels["circular"] = self._create_circular_gaussian_kernel()
            kernels["laplaciana_circular"] = (
                self._create_laplacian_of_gaussian_kernel()
            )

        return kernels

    # Cria um kernel de gabor quadrado de tamanho = kernel_size
    # Angulacao do filtro = angle_degrees
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
    
    # Cria um kernel de gabor circular (gaussiano) simetrico quadrado de tamanho = kernel_size
    # Dispersao do filtro -> proporcional a sigma
    def _create_circular_gaussian_kernel(self) -> np.ndarray:
        radius_squared = self._create_radius_squared_grid()
        kernel = np.exp(-radius_squared / (2.0 * self.gaussian_sigma**2))

        kernel /= kernel.sum()

        return kernel.astype(np.float32)

    # Cria um kernel de gabor circular (laplaciana do gaussiano) simetrico quadrado de tamanho = kernel_size
    # Dispersao do filtro -> proporcional a sigma
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
