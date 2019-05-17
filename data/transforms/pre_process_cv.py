import cv2
import data.transforms.functional_cv as vf


def transform_test(img, min_image_size, mean, std):
    img_p = vf.resize(img, min_image_size)
    img_p = vf.to_tensor(img_p) * 255
    img_p = vf.normalize(img_p, mean, std)
    return img_p, img


def load_test(filename, min_image_size=800, mean=(102.9801, 115.9465, 122.7717),
              std=(1.0, 1.0, 1.0)):
    assert isinstance(filename, str)
    img = cv2.imread(filename)
    return transform_test(img, min_image_size, mean, std)



