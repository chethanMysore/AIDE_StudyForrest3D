from torchvision import transforms as T
from torchvision.transforms import functional as F
import random


class RandomRotateTransformation:
    def __init__(self):
        self.RotationDegree = None

    def get_transform(self):
        RotationDegrees = [0, 90, 180, 270]
        self.RotationDegree = RotationDegrees[random.randint(0, 3)]
        return T.RandomRotation((self.RotationDegree, self.RotationDegree))

    def get_inverse_transform(self):
        if self.RotationDegree is None:
            print("No Transformation found")
            return
        return T.RandomRotation((self.RotationDegree * -1, self.RotationDegree * -1))


class RandomHorizontalFlipTransformation:
    def __init__(self):
        self.RotationDegree = None

    def get_transform(self):
        RotationDegrees = [0, 90, 180, 270]
        self.RotationDegree = RotationDegrees[random.randint(0, 3)]
        return T.RandomRotation((self.RotationDegree, self.RotationDegree))

    def get_inverse_transform(self):
        if self.RotationDegree is None:
            print("No Transformation found")
            return
        return T.RandomRotation((self.RotationDegree * -1, self.RotationDegree * -1))


class RandomAffineTransformation:
    def __init__(self):
        self.RotationDegree = None

    def get_transform(self):
        self.RotationDegree = random.randint(30, 90)
        return T.RandomAffine((self.RotationDegree, self.RotationDegree))
