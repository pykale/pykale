import torch
import torch.nn as nn
from create_dataloader import DTIDataset
from tdc.multi_pred import DTI
from torch_geometric.data import DataLoader
from model import DrugGCNEncoder, TargetConvEncoder, MLPDecoder
from tqdm import tqdm

TRAIN_BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
NUM_EPOCH = 100
LR = 0.005

if __name__ == "__main__":
    dataset = "DAVIS"
    data = DTI(name=dataset)
    split = data.get_split()
    train_data = DTIDataset(dataset=dataset + f"_train", root="data")
    valid_data = DTIDataset(dataset=dataset + f"_valid", root="data")
    test_data = DTIDataset(dataset=dataset + f"_test", root="data")

    train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug_encoder = DrugGCNEncoder()
    drug_encoder = drug_encoder.to(device)
    target_encoder = TargetConvEncoder()
    target_encoder = target_encoder.to(device)
    mlp_decoder = MLPDecoder()
    mlp_decoder = mlp_decoder.to(device)

    loss_fn = nn.MSELoss()
    params = list(drug_encoder.parameters()) + list(target_encoder.parameters()) + list(mlp_decoder.parameters())
    optim = torch.optim.Adam(params, lr=LR)

    for epoch in range(NUM_EPOCH):
        drug_encoder.train()
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            optim.zero_grad()
            drugs_emb = drug_encoder(x, edge_index, batch)
            targets_emb = target_encoder(data.target)
            com_emb = torch.cat((drugs_emb, targets_emb), dim=1)
            output = mlp_decoder(com_emb)
            loss = loss_fn(output, data.y.view(-1, 1).to(device))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch+1}, mse_loss: {total_loss}")
