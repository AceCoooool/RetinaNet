# TODO: modify to new version
import cv2
import data.transforms.functional_cv as vf
import data.transforms.pair_cv as T


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


def transforms_eval(cfg):
    return T.Compose([
        T.Resize(cfg.INPUT.min_size_test, cfg.INPUT.max_size_test),
        T.ToTensor(255.0),
        T.Normalize(mean=cfg.INPUT.pixel_mean, std=cfg.INPUT.pixel_std)
    ])


def transforms_train(cfg):
    return T.Compose([
        T.ColorJitter(brightness=cfg.INPUT.brightness, contrast=cfg.INPUT.contrast,
                      saturation=cfg.INPUT.saturation, hue=cfg.INPUT.hue),
        T.Resize(cfg.INPUT.min_size_train, cfg.INPUT.max_size_train),
        T.RandomHorizontalFlip(cfg.INPUT.flip_prob),
        T.ToTensor(255 if cfg.INPUT.use_255 else 1.0),
        T.Normalize(mean=cfg.INPUT.pixel_mean, std=cfg.INPUT.pixel_std)
    ])
