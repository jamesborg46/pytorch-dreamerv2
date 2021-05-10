import torch


class ImageEncodeor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 4, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 4, 2)
        self.conv4 = torch.nn.Conv2d(128, 256, 4, 2)

    def forward(self, img):
        pass

class ImageDecoder(torch.nn.Module):
    pass


