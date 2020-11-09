import torchvision.transforms as transforms
import kale.loaddata.video_data.nptransforms as nptransforms


def get_transform(kind, augment=False):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        augment (bool, optional): whether to do data augmentation (random crop and flipping). Defaults to False. (Not implemented for digits yet.)

    """

    if kind == "epic":
        transform = {
            'train': transforms.Compose([
                nptransforms.RandomCrop(size=224),
                nptransforms.RandomHorizontalFlip(),
            ]),
            'valid': transforms.Compose([
                nptransforms.CenterCrop(size=224),
            ]),
            'test': transforms.Compose([
                nptransforms.CenterCrop(size=224),
            ])
        }


    elif kind == "cifar":
        if augment:
            transform_aug = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            transform_aug = transforms.Compose([])
        transform = transforms.Compose(
            [
                transform_aug,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    else:
        raise ValueError(f"Unknown transform kind '{kind}'")
    return transform
