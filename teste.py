from pathlib import Path
from kernel_bank import KernelBank
from image import Image


kernel_bank = KernelBank()

kernel_bank.print()

path = Path("./imagem_teste.png")
img = Image(path, True)

img.show_image()

scale_1 = img.gaussian_downscale()
scale_1.show_image()

scale_2 = scale_1.gaussian_downscale()
scale_2.show_image()


