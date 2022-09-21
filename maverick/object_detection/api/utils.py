from io import BytesIO
from PIL import Image


def create_in_memory_image(image):
    file = BytesIO()
    image_f = Image.frombuffer('RGB', (image.shape[1], image.shape[0]), image, 'raw')
    image_f.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file


def create_in_memory_image_from_pil_image(image):
    file = BytesIO()
    image.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file
