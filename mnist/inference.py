import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image_To_numpy(show_info: bool=True):
    pic = Image.open('out.png')
    pix = np.array(pic)
    if show_info:
        fig = plt.figure(figsize=(4, 4))
        fig.add_subplot(1, 1,1)
        plt.tick_params(bottom=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
        plt.imshow(pix, cmap='gray_r')
    plt.show(block=True)
    return pix

if __name__ == '__main__':
    load_image_To_numpy()
