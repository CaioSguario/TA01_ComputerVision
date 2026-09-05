from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(
            self,
            path:Path | None = None,
            grey_scale:bool = True,
            height:int = 512,
            width:int = 512,
            image:np.ndarray | None = None,
        ) -> None:

        if height < 0 or width < 0:
            raise ValueError("Image width and height need to be positive integers.")

        self.height = height
        self.width = width 
        self.path = path
        self.grey_scale = grey_scale

        if image is not None:
            self.image = image

        elif Path is not None:
            self.image = self.load_image(self.path)

        else:
            raise ValueError("Either 'path' or 'image' must be provided.")


    def show_image(self, image: np.ndarray | None = None, title: str = "Image",) -> None:
        if image is None:
            image = self.image

        if image is None:
            raise ValueError("No image to display")

        plt.figure()

        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.axis("off")
        plt.show()

    def _color_to_grey(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("No image loaded.")

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def load_image(self, path:Path = None) -> np.ndarray:
        if path is not None:
            img = cv2.imread(str(path))
            self.path = path
        elif self.path is not None:
            img = cv2.imread(str(self.path))

        if img is None:
            raise FileNotFoundError(f"Image File not found.")

        # Original image dimensions
        height, width = img.shape[:2]

        # Check if image is large enough
        if height < self.height or width < self.width:
            raise ValueError(
                f"Image ({width}x{height}) is smaller than "
                f"the required size ({self.width}x{self.height})."
            )

        center_y = height // 2
        center_x = width // 2

        start_y = center_y - self.height // 2
        start_x = center_x - self.width // 2

        end_y = start_y + self.height
        end_x = start_x + self.width

        # Center crop
        img = img[
            start_y:end_y,
            start_x:end_x
        ]

        self.image = img

        if self.grey_scale:
            self._color_to_grey()

        return self.image


    def gaussian_downscale(self, 
                           image: np.ndarray | None = None, 
                           downscale=2
                        ) -> np.ndarray:

        if downscale <= 0:
            raise ValueError("Downscale must be a positive integer.")

        if image is None:
            if self.image is None:
                raise ValueError("No image loaded.")

            image = self.image

        # Gaussian blur before downscale
        sigma = downscale / 2

        image = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        height, width = image.shape[:2]

        new_width = width // downscale
        new_height = height // downscale

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        # Returns a new instance of Image
        return Image(
                    height=new_height,
                    width=new_width,
                    image=image,
                    grey_scale=self.grey_scale,
                    path=None,
                )
