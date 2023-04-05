import torch


from kale.embed.multimodal_common_fusions import Concat, MultiplicativeInteractions2Modal, LowRankTensorFusion
from kale.embed.lenet import LeNet
from kale.predict.two_layered_mlp import MLP
from kale.loaddata.avmnist_datasets import AVMNISTDataset
from trainer import Trainer

if __name__ == '__main__':
    dataset = AVMNISTDataset(data_dir='avmnist',data_size = 40)
    traindata = dataset.get_train_loader()
    validdata = dataset.get_valid_loader()
    testdata = dataset.get_test_loader()
    print("Data Loaded Successfully")

    channels = 6
    encoders = [LeNet(1, channels, 3), LeNet(1, channels, 5)]
    head = MLP(channels * 40, 100, 10)

    fusion = Concat()

    print("Model loading succesfully")
    trainer = Trainer(encoders, fusion, head, traindata, validdata,testdata, 1, optimtype=torch.optim.SGD, lr=0.01, weight_decay=0.0001)
    trainer.train()
    print("Model trained succesfully")


    print("Testing:")
    model = torch.load('best.pt')  # .cuda()
    trainer.single_test(model)
