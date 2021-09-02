# import matplotlib.image as mpimg
import os
import numpy as np
# First, load the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
# filename = "/Users/Admin/PycharmProjects/sampleGAN/data/floral/class_1_Index_0.jpeg"

# Load the image
# image = mpimg.imread(filename)

# Print out its shape
#print(image.shape)
# (10000, 784)

from PIL import Image, ImageFilter

def imagePrepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[1])
    height = float(im.size[0])
    newImage = Image.new('L',(50,50),(255))
    if width > height:
        nheight = int(round((20.0 / width * height),0))
        if (nheight == 0):
            nheight == 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((50 - nheight) / 2),0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / width * height), 0))
        if (nwidth == 0):
            nwidth == 1
        img = im.resize((20, nwidth), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((50 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(len(tva))
    print(tva)
    return tva
test = []
# directory = '/Users/Admin/PycharmProjects/sampleGAN/data/geometric/'
# for filename in os.listdir(directory):
#     if filename.endswith(".jpeg"):
test.append(imagePrepare("/Users/Admin/PycharmProjects/sampleGAN/data/geometric/class_2_Index_4.jpeg"))

a = np.array(test)
print(a.shape)