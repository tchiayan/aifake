from src.utils import get_transforms
from PIL import Image
import numpy as np

def test_get_transforms():
    transforms = get_transforms(train=True)

    # Generate dummy PIL image
    image = Image.fromarray(np.random.randint(0, 255 , (1000 , 1000 , 3)) , mode="RGB")
    transformed_image = transforms(image)

    assert transformed_image.shape == (3, 256, 256)

    transforms = get_transforms(train=False)
    transformed_image = transforms(image)

    assert transformed_image.shape == (3, 256, 256)
