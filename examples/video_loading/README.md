# Efficient Video Dataset Loading, Preprocessing, and Augmentation
Author: [Raivo Koot](https://github.com/RaivoKoot)

If you are completely unfamiliar with loading datasets in PyTorch using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`, I recommend
getting familiar with these first through [PyTorch data loading tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) or
[PyTorch dataloading examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples).
### Overview: This example demonstrates the use of `kale.loaddata.videos.VideoFrameDataset`
The VideoFrameDataset class serves to `easily`, `efficiently` and `effectively` load video samples from video datasets in PyTorch.
1) Easily because this dataset class can be used with custom datasets with minimum effort and no modification. The class merely expects the
video dataset to have a certain structure on disk and expects a .txt annotation file that enumerates each video sample. Details on this
can be found at `https://pykale.readthedocs.io/en/latest/kale.loaddata.html#kale-loaddata-videos-module`.
2) Efficiently because the video loading pipeline that this class implements is very fast. This minimizes GPU waiting time during training by eliminating input bottlenecks
that can slow down training time by several folds.
3) Effectively because the implemented sampling strategy for video frames is very strong. Video training using the entire sequence of
video frames (often several hundred) is too memory and compute intense. Therefore, this implementation samples frames evenly from the video (sparse temporal sampling)
so that the loaded frames represent every part of the video, with support for arbitrary and differing video lengths within the same dataset.
This approach has shown to be very effective and is taken from
["Temporal Segment Networks (ECCV2016)"](https://arxiv.org/abs/1608.00859) with modifications.

In conjunction with PyTorch's DataLoader, the VideoFrameDataset class returns video batch tensors of size `BATCH x FRAMES x CHANNELS x HEIGHT x WIDTH`.

For a demo, visit `main.py`.
### QuickDemo (main.py)
```python
root = os.path.join(os.getcwd(), 'demo_dataset')  # Folder in which all videos lie in a specific structure
annotation_file = os.path.join(root, 'annotations.txt')  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_INDEX)

""" DEMO 1 WITHOUT IMAGE TRANSFORMS """
dataset = VideoFrameDataset(
    root_path=root,
    annotationfile_path=annotation_file,
    num_segments=5,
    frames_per_segment=1,
    image_template='img_{:05d}.jpg',
    transform=None,
    random_shift=True,
    test_mode=False
)

sample = dataset[0]  # take first sample of dataset
frames = sample[0]   # list of PIL images
label = sample[1]    # integer label

for image in frames:
    plt.imshow(image)
    plt.title(label)
    plt.show()
    plt.pause(1)
```
![alt text](https://github.com/RaivoKoot/images/blob/main/Action_Video.jpg "Action Video")
# Table of Contents
- [Efficient Video Dataset Loading, Preprocessing, and Augmentation](#efficient-video-dataset-loading-preprocessing-and-augmentation)
    - [Overview: This example demonstrates the use of `kale.loaddata.videos.VideoFrameDataset`](#overview-this-example-demonstrates-the-use-of-kaleloaddatavideosvideoframedataset)
    - [QuickDemo (main.py)](#quickdemo-mainpy)
- [Table of Contents](#table-of-contents)
    - [1. Requirements](#1-requirements)
    - [2. Custom Dataset](#2-custom-dataset)
    - [3. Video Frame Sampling Method](#3-video-frame-sampling-method)
    - [4. Alternate Video Frame Sampling Methods](#4-alternate-video-frame-sampling-methods)
    - [5. Using VideoFrameDataset for training](#5-using-videoframedataset-for-training)
    - [6. Allowing Multiple Labels per Sample](#6-allowing-multiple-labels-per-sample)
    - [7. Conclusion](#7-conclusion)
    - [8. Acknowledgements](#8-acknowledgements)

### 1. Requirements
```
# Without these three, VideoFrameDataset will not work.
torchvision >= 0.10.0
torch >= 1.9.0
python >= 3.7,<3.10
```

### 2. Custom Dataset
(This description explains using custom datasets where each sample has a single class label. If you want to know how to
use a dataset where a sample can have more than a single class label, read this anyways and then read `6.` below)

To use any dataset, two conditions must be met.
1) The video data must be supplied as RGB frames, each frame saved as an image file. Each video must have its own folder, in which the frames of
that video lie. The frames of a video inside its folder must be named uniformly with consecutive indices such as `img_00001.jpg` ... `img_00120.jpg`, if there are 120 frames.
   Indices can start at zero or any other number and the exact file name template can be chosen freely. The filename template
   for frames in this example is "img_{:05d}.jpg" (python string formatting, specifying 5 digits after the underscore), and must be supplied to the
   constructor of VideoFrameDataset as a parameter. Each video folder must lie inside some `root` folder.
2) To enumerate all video samples in the dataset and their required metadata, a `.txt` annotation file must be manually created that contains a row for each
video clip sample in the dataset. The training, validation, and testing datasets must have separate annotation files. Each row must be a space-separated list that contains
`VIDEO_PATH START_FRAME END_FRAME CLASS_INDEX`. The `VIDEO_PATH` of a video sample should be provided without the `root` prefix of this dataset.

This example project demonstrates this using a dummy dataset inside of `demo_dataset/`, which is the `root` dataset folder of this example. When you run main.py, the dataset
folder will be automatically downloaded to this directory. The folder structure looks as follows:
```
demo_dataset
│
├───annotations.txt
├───jumping # arbitrary class folder naming
│       ├───0001  # arbitrary video folder naming
│       │     ├───img_00001.jpg
│       │     .
│       │     └───img_00017.jpg
│       └───0002
│             ├───img_00001.jpg
│             .
│             └───img_00018.jpg
│
└───running # arbitrary folder naming, no videos inside yet


```
The accompanying annotation `.txt` file contains the following rows (PATH, START_FRAME, END_FRAME, LABEL_ID)
```
jumping/0001 1 17 0
jumping/0002 1 18 0
```
Another annotations file that uses multiple clips from each video could be
```
jumping/0001 1 8 0
jumping/0001 5 17 0
jumping/0002 1 18 0
jumping/0002 10 18 0
```
(END_FRAME is inclusive)

Instantiating a VideoFrameDataset with the `root_path` parameter pointing to `demo_dataset`, the `annotationsfile_path` parameter pointing to the annotation file, and
the `imagefile_template` parameter as "img_{:05d}.jpg", is all that it takes to start using the VideoFrameDataset class.
### 3. Video Frame Sampling Method
When loading a video, only a number of its frames are loaded. They are chosen in the following way:
1. The frame index range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS even segments. From each segment, a random start-index is sampled from which FRAMES_PER_SEGMENT consecutive indices are loaded.
This results in NUM_SEGMENTS*FRAMES_PER_SEGMENT chosen indices, whose frames are loaded as PIL images and put into a list and returned when calling
`dataset[i]`.

![alt text](https://github.com/RaivoKoot/images/blob/main/Sparse_Temporal_Sampling.jpg "Sparse-Temporal-Sampling-Strategy")

### 4. Alternate Video Frame Sampling Methods
If you do not want to use sparse temporal sampling and instead want to sample a single N-frame continuous
clip from a video, this is possible. Set `NUM_SEGMENTS=1` and `FRAMES_PER_SEGMENT=N`. Because VideoFrameDataset
will chose a random start index per segment and take `NUM_SEGMENTS` continuous frames from each sampled start
index, this will result in a single N-frame continuous clip per video that starts at a random index.
An example of this is in `demo.py`.

### 5. Using VideoFrameDataset for training
As demonstrated in `main.py`, we can use PyTorch's `torch.utils.data.DataLoader` class with VideoFrameDataset to take care of shuffling, batching, and more.
To turn the lists of PIL images returned by VideoFrameDataset into tensors, the transform `kale.prepdata.video_transform.ImglistToTensor()` can be supplied
as the `transform` parameter to VideoFrameDataset. This turns a list of N PIL images into a batch of images/frames of shape `N x CHANNELS x HEIGHT x WIDTH`.
We can further chain preprocessing and augmentation functions that act on batches of images onto the end of `ImglistToTensor()`, as seen in `main.py`.

As of `torchvision 0.8.0`, all torchvision transforms can now also operate on batches of images, and they apply deterministic or random transformations
on the batch identically on all images of the batch. Because a single video-tensor (FRAMES x CHANNELS x HEIGHT x WIDTH)
has the same shape as an image batch tensor (BATCH x CHANNELS x HEIGHT x WIDTH), any torchvision transform can be used here to apply video-uniform preprocessing and augmentation.

REMEMBER:
Pytorch transforms are applied to individual dataset samples (in this case a list of PIL images of a video, or a video-frame tensor after `ImglistToTensor()`) before
batching. So, any transforms used here must expect its input to be a frame tensor of shape `FRAMES x CHANNELS x HEIGHT x WIDTH` or a list of PIL images if `ImglistToTensor()` is not used.

### 6. Allowing Multiple Labels per Sample
Your dataset labels might be more complicated than just a single label id per sample. For example, in the EPIC-KITCHENS dataset
each video clip has a verb class, noun class, and action class. In this case, each sample is associated with three label ids.
To accommodate for datasets where a sample can have N integer labels, `annotation.txt` files can be used where each row
is space separated `PATH,   FRAME_START,    FRAME_END,    LABEL_1_ID,    ...,    LABEL_N_ID`, instead of
`PATH,   FRAME_START,    FRAME_END,    LABEL_ID`. The VideoFrameDataset class
can handle this type of annotation files too, without changing anything apart from the rows in your `annotations.txt`.

The `annotations.txt` file for a dataset where multiple clip samples can come from the same video and each sample has
three labels, would have rows like `PATH,   START_FRAME,    END_FRAME,    LABEL1,    LABEL2,    LABEL3` as seen below
```
jumping/0001 1 8 0 2 1
jumping/0001 5 17 0 10 3
jumping/0002 1 18 0 5 3
running/0001 10 15 1 3 3
running/0001 5 10 1 1 0
running/0002 1 15 1 12 4
```

When you use `torch.utils.data.DataLoader` with VideoFrameDataset to retrieve your batches during
training, the dataloader then no longer returns batches as a `( (BATCHxFRAMESxHEIGHTxWIDTH) , (BATCH) )` tuple, where the second item is
just a list/tensor of the batch's labels. Instead, the second item is replaced with the tuple
`( (BATCH) ... (BATCH) )` where the first BATCH-sized list gives label_1 for the whole batch, and the last BATCH-sized
list gives label_n for the whole batch.

A demo of this can be found at the end in `main.py`. It uses the dummy dataset in directory `demo_dataset_multilabel`,
that is also automatically downloaded.

### 7. Conclusion
A proper code-based explanation on how to use VideoFrameDataset for training is provided in `main.py`

### 8. Acknowledgements
We thank the authors of TSN for their [codebase](https://github.com/yjxiong/tsn-pytorch), from which we took VideoFrameDataset and adapted it
for general use and compatibility.
```
@InProceedings{wang2016_TemporalSegmentNetworks,
    title={Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
    author={Limin Wang and Yuanjun Xiong and Zhe Wang and Yu Qiao and Dahua Lin and
            Xiaoou Tang and Luc {Val Gool}},
    booktitle={The European Conference on Computer Vision (ECCV)},
    year={2016}
}
```
