import sys
from PIL import Image

sys.path.append('../torchvision/')
from transforms import *

# Very simple script using new functionality that should become a unit test
# at some point.

test_img_path = './assets/grace_hopper_517x606.jpg'
test_img = Image.open(test_img_path)
test_img.show()

transform = Compose([
    RandomHorizontalFlip(0.2),
    RandomVerticalFlip(0.2),
    RandomShear(),
    RandomRotate(),
    RandomGammaIntensity(),
    GaussianNoise()
])

import time
start = time.time()
for i in range(5):
    transform(test_img).show()
print(time.time() - start)
