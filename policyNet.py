import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from time import strftime, localtime
import numpy as np
import gzip
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from collections import defaultdict, OrderedDict
from rdchiral.main import rdchiralRunText, rdchiralRun


def preprocess(X, fp_dim):
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim, n_rules)

    def forward(self, x, y=None, loss_fn=nn.CrossEntropyLoss()):
        x = self.fc3(self.dropout1(self.bn1(self.fc1(x))))
        if y is not None:
            return loss_fn(x, y)
        else:
            return x


def load_parallel_model(state_path, template_rule_path, fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule = l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules), fp_dim=fp_dim)
    checkpoint = torch.load(state_path, map_location='cpu')
    rollout.load_state_dict(checkpoint)
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint.items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    #rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule


class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr, [-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1)
        preds = preds.cpu()
        probs, idx = torch.topk(preds, k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i, rule in enumerate(rule_k):
            try:
                out1 = rdchiralRunText(rule, x)
                if len(out1) == 0:
                    continue
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    templates.append(rule)
            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                pass
        if len(reactants) == 0:
            return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))
        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]
        return {'reactants': reactants,
                'scores': scores,
                'template': templates}


def top_k_acc(preds, gt, k=1):
    probs, idx = torch.topk(preds, k=k)
    idx = idx.cpu().numpy().tolist()
    gt = gt.cpu().numpy().tolist()
    num = preds.size(0)
    correct = 0
    for i in range(num):
        if gt[i] in idx[i]:
            correct += 1
    return correct, num


class OneStepDataset(Dataset):
    def __init__(self, data):
        self.x = data['smiles']
        self.y = data['template']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_fp = np.unpackbits(self.x[item])
        return x_fp, self.y[item]


def dataset_iterator(data, batch_size=1024, shuffle=True, num_workers=4):
    dataset = OneStepDataset(data)
    def collate_fn(batch):
        X, y = zip(*batch)
        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


def train_one_epoch(net, train_loader, optimizer, it, device):
    losses = []
    net.train()
    fr = open('loss.txt', 'a')
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss_v = net(X_batch, y_batch)
        loss_v = loss_v.mean()
        loss_v.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss_v.item())
        fr.write(str(loss_v.item()) + '\n')
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
    fr.close()
    return losses


def eval_one_epoch(net, val_loader, device):
    net.eval()
    eval_top1_correct, eval_top1_num = 0, 0
    eval_top10_correct, eval_top10_num = 0, 0
    eval_top50_correct, eval_top50_num = 0, 0
    loss = 0.0
    for X_batch, y_batch in tqdm(val_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_hat = net(X_batch)
            loss += F.cross_entropy(y_hat,y_batch).item()
            top_1_correct, num1 = top_k_acc(y_hat, y_batch, k=1)
            top_10_correct, num10 = top_k_acc(y_hat, y_batch, k=10)
            top_50_correct, num50 = top_k_acc(y_hat, y_batch, k=50)
            eval_top1_correct += top_1_correct
            eval_top1_num += num1
            eval_top10_correct += top_10_correct
            eval_top10_num += num10
            eval_top50_correct += top_50_correct
            eval_top50_num += num50
    val_1 = eval_top1_correct/eval_top1_num
    val_10 = eval_top10_correct/eval_top10_num
    val_50 = eval_top50_correct/eval_top50_num
    loss = loss / (len(val_loader.dataset))
    return val_1, val_10, val_50, loss


def train(net, dataTrain, dataTest, lr=0.001, batch_size=16, epochs=100, wd=0, saved_model='./model/saved_states'):
    it = trange(epochs)
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    device = 'cpu'
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    train_loader = dataset_iterator(dataTrain, batch_size=batch_size)
    val_loader = dataset_iterator(dataTest, batch_size=batch_size, shuffle=False)
    best = -1
    for e in it:
        train_one_epoch(net, train_loader, optimizer, it, device)
        val_1, val_10, val_50, loss = eval_one_epoch(net, val_loader, device)
        scheduler.step(loss)
        if best < val_1:
            best = val_1
            state = net.state_dict()
            time_stamp = strftime("%Y-%m-%d_%H:%M:%S", localtime())
            save_path = saved_model + "_" + time_stamp + '.ckpt'
            torch.save(state, save_path)
        line = "\nTop 1: {}  ==> Top 10: {} ==> Top 50: {}, validation loss ==> {}".format(val_1, val_10, val_50, loss)
        fr = open('result.txt', 'a')
        fr.write(line)
        fr.close()
        print(line)


def train_mlp(batch_size=1024, lr=0.001, epochs=100, weight_decay=0, dropout_rate=0.3, saved_model='./model/saved_rollout_state'):
    with gzip.open('./prepare_data/uspto_template.pkl.gz', 'rb') as f:
        templates = pickle.load(f)
    num_of_rules = len(templates)
    rollout = RolloutPolicyNet(n_rules=num_of_rules, dropout_rate=dropout_rate)
    print('mlp model training...')
    train_path = './prepare_data/policyTrain.pkl.gz'
    with gzip.open(train_path, 'rb') as f:
        trainData = pickle.load(f)
    test_path = './prepare_data/policyTest.pkl.gz'
    with gzip.open(test_path, 'rb') as f:
        testData = pickle.load(f)
    print('Training size:', len(trainData['smiles']))
    train(rollout, dataTrain=trainData, dataTest=testData, batch_size=batch_size, lr=lr, epochs=epochs, wd=weight_decay, saved_model=saved_model)


def loadPolicyModel():
    with gzip.open('./prepare_data/uspto_template.pkl.gz', 'rb') as f:
        templates = pickle.load(f)
    num_of_rules = len(templates)
    rollout = RolloutPolicyNet(n_rules=num_of_rules, dropout_rate=0.4)
    rollout.load_state_dict(torch.load('./model/saved_rollout_state_2023-01-09_04:38:49.ckpt'))
    rollout.eval()
    return rollout


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--model_folder', default='./model', type=str, help='specify where to save the trained models')
    parser.add_argument('--batch_size', default=2056, type=int, help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float, help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.01, type=float, help="specify the learning rate")
    args = parser.parse_args()
    model_folder = args.model_folder
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    lr = args.learning_rate
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    train_mlp(batch_size=batch_size, lr=lr, dropout_rate=dropout_rate)
