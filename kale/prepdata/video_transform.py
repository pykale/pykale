import torch
from torchvision import transforms


def get_transform(kind, modality):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        modality (string): image type (RGB or Optical Flow)
    """

    if kind in ["epic", "gtea", "adl", "kitchen"]:
        transform = dict()
        if modality == "rgb":
            transform = {
                "train": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.RandomCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
            }
        elif modality == "flow":
            transform = {
                "train": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.RandomCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
                "valid": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        # Stack(),
                        ImglistToTensor(),
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=224),
                        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                        TensorPermute(),
                    ]
                ),
            }

    else:
        raise ValueError(f"Unknown transform kind '{kind}'")
    return transform


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``kale.loaddata.videos.VideoFrameDataset``.
    """

    def forward(self, img_list):
        """
        For RGB input, Converts each PIL image in a list to a torch Tensor and stacks them into a single tensor.
        For flow input, after the previous step, reshapes the tensor to combine the x(u)_img, y(v)_img. For example,
        assuming flow tensor size is [16, 1, 224, 224], the frame order is [frame 1_x, frame 1_y, frame 2_x,
        frame 2_y, frame 3_x, ..., frame 8_x, frame 8_y], reshapes it to [8, 2, 224, 224] to create [[frame 1_x,
        frame 1_y], [frame 2_x, frame 2_y], [frame 3_x, ..., [frame 8_x, frame 8_y]] to combine x and y. Now Frame
        num is 8, channel num is 2.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size `` NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        img_t = torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
        if img_list[0].mode == "RGB":
            return img_t
        elif img_list[0].mode == "L":
            img_f = torch.reshape(img_t, shape=(img_t.shape[0] // 2, 2, img_t.shape[2], img_t.shape[3]))
            return img_f
        else:
            raise RuntimeError("Image modality is not in [rgb, flow].")


class TensorPermute(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) to
    a torch.FloatTensor of shape (CHANNELS x NUM_IMAGES x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(1, 0, 2, 3).contiguous()
