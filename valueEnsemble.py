import os
import numpy as np
import torch
import pickle
import torch.nn as nn
from GraphEncoder import GraphModel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def unpack_fps(packed_fps):
    packed_fps = packed_fps.astype(np.uint8)
    shape = (*(packed_fps.shape[:-1]), -1)
    fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])), axis=-1).astype(np.float32).reshape(shape)
    return fps


class ValueEnsemble(nn.Module):
    def __init__(self, fp_dim, latent_dim, dropout_rate):
        super(ValueEnsemble, self).__init__()
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.graphNN = GraphModel(input_dim=fp_dim, feature_dim=latent_dim, hidden_dim=latent_dim)
        self.layers = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, fps, mask):
        x = self.graphNN(fps, mask)
        x = self.layers(x)
        return x


class ConsistencyDataset(Dataset):
    def __init__(self, data):
        self.reaction_costs = data['reaction_costs']
        self.target_values = data['target_values']
        self.reactant_fps = data['reactant_fps']
        self.reactant_masks = data['reactant_masks']

    def __len__(self):
        return len(self.reaction_costs)

    def __getitem__(self, item):
        reaction_cost = self.reaction_costs[item]
        target_value = self.target_values[item]
        reactant_fps = np.zeros((5, 2048), dtype=np.float32)
        reactant_fps[:3, :] = unpack_fps(np.array(self.reactant_fps[item]))
        reactant_masks = np.zeros(5, dtype=np.float32)
        reactant_masks[:3] = self.reactant_masks[item]
        return reaction_cost, target_value, reactant_fps, reactant_masks


class FittingDatasetTest(Dataset):
    def __init__(self, data):
        self.fps = data['fps']
        self.values = data['values']
        self.masks = data['masks']

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.fps[item], self.values[item], self.masks[item]



class Trainer:
    def __init__(self, model, n_epochs, lr, batch_size, model_folder, device, test_epoch):
        self.batch_size = batch_size
        file = './data/train_consistency.pkl'
        with open(file, 'rb') as f:
            self.train_consistency_data = pickle.load(f)
        self.train_consistency = ConsistencyDataset(self.train_consistency_data)
        self.train_consistency_loader = DataLoader(self.train_consistency, batch_size=self.batch_size, shuffle=True)
        self.train_consistency_iter = iter(self.train_consistency_loader)
        file = './data/train_fitting.pkl'
        with open(file, 'rb') as f:
            self.train_fitting_data = pickle.load(f)
        self.num_fitting_mols = len(self.train_fitting_data['values'])
        print('Train Data Loaded')
        file = './data/val_consistency.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.val_consistency = ConsistencyDataset(data)
        self.val_consistency_loader = DataLoader(self.val_consistency, batch_size=self.batch_size, shuffle=False)
        file = './val_dataset/val_fitting.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.val_fitting = FittingDatasetTest(data)
        self.val_fitting_loader = DataLoader(self.val_fitting, batch_size=self.batch_size, shuffle=False)
        print('Validation Data Loaded.')
        self.n_epochs = n_epochs
        self.lr = lr
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(device)
        self.test_epoch = test_epoch
        self.fitting_criterion = nn.MSELoss(reduction='none')
        self.num_mols = 1
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def sample(self, num_mols, num_samples):
        fps = np.zeros((num_samples, 5, 2048), dtype=np.float32)
        values = []
        masks = np.ones((num_samples, 5))
        masks[:, num_mols:] = 0
        for n in range(num_samples):
            index = np.random.choice(self.num_fitting_mols, num_mols, replace=False)
            fp = [self.train_fitting_data['fps'][i] for i in index]
            value = np.sum([self.train_fitting_data['values'][i] for i in index])
            fps[n, :num_mols, :] = unpack_fps(np.array(fp))
            values.append(value)
        data = {
            'fps': list(fps.astype(np.float32)),
            'values': list(np.array(values).astype(np.float32)),
            'masks': list(masks.astype(np.float32))
        }
        return data

    def _pass(self, consistency_data, fitting_data):
        self.optim.zero_grad()
        fps, values, masks = fitting_data['fps'], fitting_data['values'], fitting_data['masks']
        fps = torch.FloatTensor(fps).to(self.device)
        values = torch.FloatTensor(values).to(self.device).reshape(-1)
        masks = torch.FloatTensor(masks).to(self.device)
        weight = torch.FloatTensor(np.array([1 / self.num_mols] * len(values)).astype(np.float32)).to(device)
        v_pred = self.model(fps, masks)
        #fitting_loss = F.mse_loss(v_pred.reshape(-1), values)
        fitting_loss = self.fitting_criterion(v_pred.reshape(-1), values)
        fitting_loss = (weight * fitting_loss).mean()
        reaction_costs, target_values, reactant_fps, reactant_masks = consistency_data
        reaction_costs = torch.FloatTensor(reaction_costs).to(self.device).reshape(-1)
        target_values = torch.FloatTensor(target_values).to(self.device).reshape(-1)
        reactant_fps = torch.FloatTensor(reactant_fps).to(self.device)
        reactant_masks = torch.FloatTensor(reactant_masks).to(self.device)
        r_values = self.model(reactant_fps, reactant_masks)
        r_gap = - r_values - reaction_costs + target_values + 7.
        r_gap = torch.clamp(r_gap, min=0)
        consistency_loss = (r_gap ** 2).mean()
        loss = fitting_loss + consistency_loss
        loss.backward()
        self.optim.step()
        fr = open('loss.txt', 'a')
        line = str(loss.item()) + '\t' + str(fitting_loss.item()) + '\t' + str(consistency_loss.item()) + '\n'
        fr.write(line)
        fr.close()

    def eval(self):
        self.model.eval()
        consistency_loss = []
        fitting_loss = []
        for batch in self.val_consistency_loader:
            reaction_costs, target_values, reactant_fps, reactant_masks = batch
            reaction_costs = torch.FloatTensor(reaction_costs).to(self.device).reshape(-1)
            target_values = torch.FloatTensor(target_values).to(self.device).reshape(-1)
            reactant_fps = torch.FloatTensor(reactant_fps).to(self.device)
            reactant_masks = torch.FloatTensor(reactant_masks).to(self.device)
            r_values = self.model(reactant_fps, reactant_masks)
            r_gap = - r_values - reaction_costs + target_values + 7.
            r_gap = torch.clamp(r_gap, min=0)
            consistency_loss.append((r_gap ** 2).mean().item())
        for batch in self.val_fitting_loader:
            fps, values, masks = batch
            fps = torch.FloatTensor(fps).to(self.device)
            values = torch.FloatTensor(values).to(self.device).reshape(-1)
            masks = torch.FloatTensor(masks).to(self.device)
            v_pred = self.model(fps, masks)
            fitting_loss.append(F.mse_loss(v_pred.reshape(-1), values).item())
        fr = open('test.txt', 'a')
        fr.write(str(np.mean(consistency_loss)) + '\t' + str(np.mean(fitting_loss)) + '\n')
        fr.close()

    def sampleMols(self):
        randomNumber = np.random.rand()
        if randomNumber < 0.69:
            return 1
        if randomNumber < 0.94:
            return 2
        if randomNumber < 0.99:
            return 3
        if randomNumber < 0.995:
            return 4
        return 5

    def train(self):
        self.model.train()
        for epoch in range(self.n_epochs):
            consistency_batch = next(self.train_consistency_iter)
            if len(consistency_batch) < self.batch_size:
                self.train_consistency_loader = DataLoader(self.train_consistency, batch_size=self.batch_size, shuffle=True)
                self.train_consistency_iter = iter(self.train_consistency_loader)
                consistency_batch = next(self.train_consistency_iter)
            num_mols = self.sampleMols()
            self.num_mols = num_mols
            fitting_batch = self.sample(num_mols, self.batch_size)
            self._pass(consistency_batch, fitting_batch)
            if (epoch + 1) % self.test_epoch == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % (epoch + 1)
                torch.save(self.model.module.state_dict(), save_file)
                self.eval()
                self.model.train()


if __name__ == '__main__':
    n_epochs = 1000000000
    lr = 0.001
    batch_size = 1024
    model_folder = './model'
    device = 'cuda:0'
    test_epoch = 100
    model = ValueEnsemble(2048, 128, dropout_rate=0.1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    trainer = Trainer(model, n_epochs, lr, batch_size, model_folder, device, test_epoch)
    trainer.train()

