import os

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms

from kale.loaddata.videos import VideoFrameDataset
from kale.prepdata.video_transform import ImglistToTensor
from kale.utils.download import download_file_gdrive

"""
Ignore this function and look at "main" below.
"""


def plot_video(rows, cols, frame_list, plot_width, plot_height):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
        axes_pad=0.3,  # pad between axes in inch.
    )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.show()


if __name__ == "__main__":
    """
    This demo uses the dummy dataset inside of the folder "demo_dataset".
    It is structured just like a real dataset would need to be structured.

    TABLE OF CODE CONTENTS:
    1. Minimal demo without image transforms
    2. Minimal demo without sparse temporal sampling for single continuous frame clips, without image transforms
    3. Demo with image transforms
    4. Demo 3 continued with PyTorch dataloader
    5. Demo of using a dataset where samples have multiple separate class labels

    """
    download_file_gdrive(
        id="1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga",
        output_directory=os.path.join(os.getcwd()),
        output_file_name="demo_datasets.zip",
        file_format=".zip",
    )

    videos_root = os.path.join(os.getcwd(), "demo_dataset")
    annotation_file = os.path.join(videos_root, "annotations.txt")

    """ DEMO 1 WITHOUT IMAGE TRANSFORMS """
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template="img_{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
    )

    sample = dataset[0]
    frames = sample[0]  # list of PIL images
    label = sample[1]  # integer label

    plot_video(rows=1, cols=5, frame_list=frames, plot_width=15.0, plot_height=3.0)

    """ DEMO 2 SINGLE CONTINUOUS FRAME CLIP INSTEAD OF SAMPLED FRAMES, WITHOUT TRANSFORMS """
    # If you do not want to use sparse temporal sampling, and instead
    # want to just load N consecutive frames starting from a random
    # start index, this is easy. Simply set NUM_SEGMENTS=1 and
    # FRAMES_PER_SEGMENT=N. Each time a sample is loaded, N
    # frames will be loaded from a new random start index.
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=1,
        frames_per_segment=9,
        imagefile_template="img_{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
    )

    sample = dataset[1]
    frames = sample[0]  # list of PIL images
    label = sample[1]  # integer label

    plot_video(rows=3, cols=3, frame_list=frames, plot_width=10.0, plot_height=5.0)

    """ DEMO 3 WITH TRANSFORMS """
    # As of torchvision 0.8.0, torchvision transforms support batches of images
    # of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
    # transformations on the batch identically on all images of the batch. Any torchvision
    # transform for image augmentation can thus also be used  for video augmentation.
    preprocess = transforms.Compose(
        [
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.Resize(299),  # image batch, resize smaller edge to 299
            transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template="img_{:05d}.jpg",
        transform=preprocess,
        random_shift=True,
        test_mode=False,
    )

    sample = dataset[1]
    frame_tensor = sample[0]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
    label = sample[1]  # integer label

    print("Video Tensor Size:", frame_tensor.size())

    """
    Denormalize is just for visualization purposes, to undo the transforms applied
    to the list of frames of a video.
    """

    def denormalize(video_tensor):
        """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        """
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        return (inverse_normalize(video_tensor) * 255.0).type(torch.uint8).permute(0, 2, 3, 1).numpy()

    frame_tensor = denormalize(frame_tensor)
    plot_video(rows=1, cols=5, frame_list=frames, plot_width=15.0, plot_height=3.0)

    """ DEMO 3 CONTINUED: DATALOADER """
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
    )

    for epoch in range(10):
        for video_batch, labels in dataloader:
            """
            Insert Training Code Here
            """
            print(labels)
            print("\nVideo Batch Tensor Size:", video_batch.size())
            print("Batch Labels Size:", labels.size())
            break
        break

    """ DEMO 5: SAMPLES WITH MULTIPLE LABELS """
    """
    Apart from supporting just a single label per sample, VideoFrameDataset also supports multi-label samples,
    where a sample can be associated with more than just one label. EPIC-KITCHENS, for example, associates a
    noun, verb, and action with each video clip. To support this, instead of each row in annotations.txt
    being (VIDEO_PATH, START_FRAME, END_FRAME, LABEL_ID), each row can also be
    (VIDEO_PATH, START_FRAME, END_FRAME, LABEL_1_ID, ..., LABEL_N_ID). An example of this can be seen in the
    directory `demo_dataset_multilabel`.

    Each sample returned by VideoFrameDataset is then ((FRAMESxCHANNELSxHEIGHTxWIDTH), (LABEL_1, ..., LABEL_N)).
    When paired with the `torch.utils.data.DataLoader`, instead of yielding each batch as
    ((BATCHxFRAMESxCHANNELSxHEIGHTxWIDTH), (BATCH)) where the second tuple item is the labels of the batch,
    `torch.utils.data.DataLoader` returns a batch as ((BATCHxFRAMESxCHANNELSxHEIGHTxWIDTH), ((BATCH),...,(BATCH))
    where the second tuple item is itself a tuple, with N BATCH-sized tensors of labels, where N is the
    number of labels assigned to each sample.
    """
    videos_root = os.path.join(os.getcwd(), "demo_dataset_multilabel")
    annotation_file = os.path.join(videos_root, "annotations.txt")

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template="img_{:05d}.jpg",
        transform=preprocess,
        random_shift=True,
        test_mode=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=3, shuffle=True, num_workers=2, pin_memory=True
    )

    print("\nMulti-Label Example")
    for epoch in range(10):
        for batch in dataloader:
            """
            Insert Training Code Here
            """
            video_batch, (labels1, labels2, labels3) = batch

            print("Video Batch Tensor Size:", video_batch.size())
            print("Labels1 Size:", labels1.size())  # == batch_size
            print("Labels2 Size:", labels2.size())  # == batch_size
            print("Labels3 Size:", labels3.size())  # == batch_size

            break
        break
