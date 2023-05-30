import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, r2_score
import os

class ts_dataset(Dataset):
    def __init__(self, ts_data, seq_len):
        super(ts_dataset, self).__init__()
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.sample_num = ts_data.shape[1] - self.seq_len + 1

    def __getitem__(self, idx):
        return self.ts_data[:, idx: idx+self.seq_len, :]

    def __len__(self):
        return self.sample_num


class CNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.bn1 = nn.BatchNorm1d(in_dim)  
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, 5) 
        self.pool1 = nn.MaxPool1d(2, 2)  
        self.bn2 = nn.BatchNorm1d(hidden_dim)   
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 5)  
        self.pool2 = nn.MaxPool1d(2, 2)  
        self.bn3 = nn.BatchNorm1d(hidden_dim)   
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 5)  
        self.pool3 = nn.MaxPool1d(2, 2) 
    def forward(self, x: torch.Tensor):
        batch, node, window_size, in_dim= x.shape
    
        x = torch.cat(torch.split(x, 1, dim=1), dim=0).transpose(-1, -2).squeeze(1)
        
        x = torch.relu(self.pool1(self.conv1(self.bn1(x))))
        x = torch.relu(self.pool2(self.conv2(self.bn2(x))))
        x = torch.relu(self.pool3(self.conv3(self.bn3(x))))
        
        x = x.squeeze(-1)
        
        x = torch.stack(torch.split(x, batch, dim=0), dim=1)

        return x


class Gumbel_Generator(nn.Module):
    def __init__(self, node_num = 8, temp = 10, temp_drop_frac = 0.9999):
        super(Gumbel_Generator, self).__init__()
        

        self.params = nn.Parameter(torch.rand(node_num, node_num, 2))
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
        

    def drop_temperature(self):
        self.temperature = self.temperature * self.temp_drop_frac

    def sample(self, eps=1e-20):
        logp = F.softmax(self.params, -1).log()
        u = torch.rand_like(logp)
        y = logp - (- (u + eps).log() + eps).log()
        y_t = y / self.temperature
        y_t_softmax = F.softmax(y_t, dim=-1)[:,:,1]
            
        return y_t_softmax
    
    def get_p(self):
        return F.softmax(self.params, -1)[:,:,1]


class BaseGumbelGraphNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(BaseGumbelGraphNetwork, self).__init__()

        self.node2edge = torch.nn.Linear(2*in_dim, hidden_dim)
        self.edge2edge = torch.nn.Linear(hidden_dim, hidden_dim)
        self.edge2node = torch.nn.Linear(hidden_dim, hidden_dim)
        self.node2node = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output1 = torch.nn.Linear(in_dim+hidden_dim, hidden_dim)
        self.output2 = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, input, adj):

        batch, node, in_dim = input.shape
        tmp_input = input
        adjs = adj[None, :, :, None]
 
        i_node = tmp_input.unsqueeze(2).repeat(1, 1, node, 1)

        j_node = i_node.transpose(1, 2)
        ij_node = torch.cat((i_node, j_node), -1)
        
        node2edge = F.relu(self.node2edge(ij_node))

        edge2edge = F.relu(self.edge2edge(node2edge))

        edge = adjs * edge2edge

        sum_edge = torch.sum(edge, 2)
        edge2node = F.relu(self.edge2node(sum_edge))
        out = F.relu(self.node2node(edge2node))
        out = torch.cat((input, out), dim = -1)
        out = torch.relu(self.output1(out))
        out = self.output2(out)

        return out


class ODEGumbelGraphNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, delta_t):
        super(ODEGumbelGraphNetwork, self).__init__()
        self.cnn = CNN(in_dim, hidden_dim)
        self.f = BaseGumbelGraphNetwork(hidden_dim, hidden_dim, in_dim)
        self.delta_t = delta_t
        
    def forward(self, x, adj):

        cnn_out = self.cnn(x)
       
        f = self.f(cnn_out, adj)
        
        out = f * self.delta_t + x[:,:,-1,:]
        return out


class SDEGumbelGraphNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, delta_t):
        super(SDEGumbelGraphNetwork, self).__init__()
        self.cnn = CNN(in_dim, hidden_dim)
        self.f = BaseGumbelGraphNetwork(hidden_dim, hidden_dim, in_dim)
        self.g = BaseGumbelGraphNetwork(hidden_dim, hidden_dim, in_dim)
        self.delta_t = delta_t
        self.sqrt_delta_t = self.delta_t ** (0.5)
        
    def forward(self, x, adj):

        cnn_out = self.cnn(x)
        

        f = self.f(cnn_out, adj)
        g = self.g(cnn_out, adj)
        
        out = f * self.delta_t + g * self.sqrt_delta_t * torch.randn_like(g, device=g.device) + x[:,:,-1,:]
        return out


class DynSeq2Seq(nn.Module):
    def __init__(self, base_dyn_learner) -> None:
        super(DynSeq2Seq, self).__init__()
        self.base_dyn_learner = base_dyn_learner
        
    def forward(self, x, adj, step):
        batch, node, window_size, in_dim = x.shape
        seq_len = window_size + step
        outputs = torch.empty(batch, node, seq_len, in_dim, device=x.device)    
        outputs[:,:,:window_size,:] = x.clone()
        
        output = x.clone()
        for t in range(step):
            output[:,:,-1,:] = self.base_dyn_learner(output, adj)
            outputs[:,:,t+window_size,:] = output[:,:,-1,:]
            output[:,:,:-1,:] = outputs[:,:,t+1:t+window_size,:]
            
        predict = outputs[:,:,window_size:,:]
        
        return predict


class NetSeq2Seq(nn.Module):
    def __init__(self, base_net_generator) -> None:
        super(NetSeq2Seq, self).__init__()
        self.base_net_generator = base_net_generator
        
    def forward(self, x, dyn_learner, step):
        batch, node, window_size, in_dim = x.shape
        seq_len = window_size + step
        adj = self.sample()
        self.base_net_generator.drop_temperature()
        outputs = torch.empty(batch, node, seq_len, in_dim, device=x.device)    
        outputs[:,:,:window_size,:] = x.clone()
        
        output = x.clone()
        for t in range(step):
            output[:,:,-1,:] = dyn_learner(output, adj)
            outputs[:,:,t+window_size,:] = output[:,:,-1,:]
            output[:,:,:-1,:] = outputs[:,:,t+1:t+window_size,:]
            
        predict = outputs[:,:,window_size:,:]
        
        return predict

    def sample(self):
        return self.base_net_generator.sample()


loss_fn = F.mse_loss


def compute_metrics(target, predict):
    mse = ((predict - target) ** 2).mean().item()
    mae = (predict - target).abs().mean().item()
    return mse, mae


def split_input_target(x, window_size):

    input = x[:,:,:window_size,:]
    target = x[:,:,window_size:,:]
    return input, target


def save_compare_results(input, target, ode_predict, sde_predict, path):
    sample_num, node_num, window_size, dim = input.shape
    step = target.shape[2]
    seq_len = window_size + step
    subpic_num = node_num * dim
    real_value = torch.cat([input, target], dim=-2).cpu().detach().numpy()
    ode_value = torch.cat([input, ode_predict], dim=-2).cpu().detach().numpy()
    sde_value = torch.cat([input, sde_predict], dim=-2).cpu().detach().numpy()

    for sample_idx in range(sample_num):
        file_name = os.path.join(path, str(sample_idx)+'.jpg')
        plt.figure(figsize=(30, 6*node_num))
        for sub_pic_idx in range(subpic_num):
            node_idx = sub_pic_idx // 2
            dim_idx = sub_pic_idx % 2
            dim_name = 'Re' if dim_idx == 0 else 'Im'
            title = 'node_' + str(node_idx + 1) + '_' + dim_name
            plt.subplot(node_num, dim, sub_pic_idx+1, title=title)
            plt.plot(range(seq_len), sde_value[sample_idx, node_idx, :, dim_idx], label='sde_model_predict')
            plt.plot(range(seq_len), ode_value[sample_idx, node_idx, :, dim_idx], label='ode_model_predict')
            plt.plot(range(seq_len), real_value[sample_idx, node_idx, :, dim_idx], label='real')   
            plt.ylim(-3,3)
            plt.legend(loc=1)
        plt.savefig(file_name)
        
    ode_mae = torch.abs(ode_predict - target).flatten(1).mean(dim=1).cpu().numpy()
    ode_mse = ((ode_predict - target) ** 2).flatten(1).mean(dim=1).cpu().numpy()
    sde_mae = torch.abs(sde_predict - target).flatten(1).mean(dim=1).cpu().numpy()
    sde_mse = ((sde_predict - target) ** 2).flatten(1).mean(dim=1).cpu().numpy()
    pd.DataFrame(
            {'idx': range(sample_num),
            'ode_mse': ode_mse,
            'ode_mae': ode_mae,
            'sde_mse': sde_mse,
            'sde_mae': sde_mae}).to_csv(os.path.join(path, 'results.csv'))

def epoch_log(summery, train_ode_mse, train_ode_mae, train_sde_mse, train_sde_mae, test_ode_mse, test_ode_mae, test_sde_mse, test_sde_mae, epoch_idx, iter_num):
    summery.add_scalars('epoch_train/mse', {'ode': train_ode_mse, 'sde': train_sde_mse}, epoch_idx)
    summery.add_scalars('epoch_train/mae', {'ode': train_ode_mae, 'sde': train_sde_mae}, epoch_idx)
    summery.add_scalars('epoch_test/mse', {'ode': test_ode_mse, 'sde': test_sde_mse}, epoch_idx)
    summery.add_scalars('epoch_test/mae', {'ode': test_ode_mae, 'sde': test_sde_mae}, epoch_idx)
    
    if epoch_idx % iter_num == iter_num - 1:
        iter_idx = epoch_idx // iter_num
        summery.add_scalars('iter_train/mse', {'ode': train_ode_mse, 'sde': train_sde_mse}, iter_idx)
        summery.add_scalars('iter_train/mae', {'ode': train_ode_mae, 'sde': train_sde_mae}, iter_idx)
        summery.add_scalars('iter_test/mse', {'ode': test_ode_mse, 'sde': test_sde_mse}, iter_idx)
        summery.add_scalars('iter_test/mae', {'ode': test_ode_mae, 'sde': test_sde_mae}, iter_idx)


def save_model(dyn_learner, net_learner, epoch_idx, path):
    torch.save(dyn_learner.state_dict(), os.path.join(path, str(epoch_idx)+'_dyn_learner.pt'))
    torch.save(net_learner.state_dict(), os.path.join(path, str(epoch_idx)+'_net_learner.pt'))


def path_check(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def train_s_dyn_learner(s_dyn_learner, s_net_learner, opt_s_dyn_learner, data, window_size):

    s_dyn_learner.train()
    s_net_learner.eval()
    s_dyn_learner.zero_grad()
    
    input, target = split_input_target(data, window_size)
    step = target.shape[2]
    adj = s_net_learner.sample()
    
    predict = s_dyn_learner(input, adj, step)
    loss = loss_fn(predict, target)
        
    loss.backward()
    opt_s_dyn_learner.step()
    
    mse, mae = compute_metrics(target, predict)
    return mse, mae


def train_s_net_learner(s_dyn_learner, s_net_learner, opt_s_net_learner, data, window_size):

    s_dyn_learner.eval()
    s_net_learner.train()
    s_net_learner.zero_grad()
    
    input, target = split_input_target(data, window_size)
    step = target.shape[2]
    
    predict = s_net_learner(input, s_dyn_learner.base_dyn_learner, step)
    loss = loss_fn(predict, target)
        
    loss.backward()
    opt_s_net_learner.step()
    
    mse, mae = compute_metrics(target, predict)
    return mse, mae


def val_s_dyn_learner(s_dyn_learner, s_net_learner, data, window_size):

    s_dyn_learner.eval()
    s_net_learner.eval()

    input, target = split_input_target(data, window_size)
    step = target.shape[2]
    adj = s_net_learner.sample()
    
    predict = s_dyn_learner(input, adj, step)
    
    mse, mae = compute_metrics(target, predict)
    return mse, mae


node_num = 8

window_size = 36

step = 36

seq_len = window_size + step


batch_size = 1024
shuffle = True
num_workers = 4


device = 'cuda:0'

root_path = '/home/mathcenter2022/Documents/lyf/hw_ggn'

data_file = os.path.join(root_path, 'dianxinshuju.csv')

results_path = os.path.join(root_path, 'results')  

model_save_path = os.path.join(root_path, 'models') 

summery_path = os.path.join(root_path, 'run')


in_dim = 2
hidden_dim = 64
delta_t = 0.002   
temp = 10
temp_drop_frac = 0.9999


dyn_lr = 0.001
net_lr = 0.0001
train_pct = 0.9
dyn_step = 12
net_step = 3


iter_step = dyn_step + net_step
iter_num = 1000
epoch_num = iter_step * iter_num

tmp_data = pd.read_csv('./dianxinshuju.csv')

tmp_cols_res = ['1', '2', '3', '4', '5', '6', '7', '8']
tmp_cols_ims = ['1.1', '2.1', '3.1', '4.1', '5.1', '6.1', '7.1', '8.1']
for tmp_col_im in tmp_cols_ims:
    tmp_data.loc[:, tmp_col_im] = [float(im_str[:-1]) for im_str in tmp_data.loc[:, tmp_col_im].values]
tmp_data = np.stack(
    [tmp_data.loc[:, tmp_cols_res].values, 
     tmp_data.loc[:, tmp_cols_res].values], axis=-1)
tmp_data = tmp_data.transpose(1,0,2).astype(np.float32)
tmp_data = np.ascontiguousarray(tmp_data)

dataset = ts_dataset(tmp_data, seq_len)

train_num = int(len(dataset) * train_pct)
test_num = len(dataset) - train_num
train_dataset, test_dataset = Subset(dataset, list(range(len(dataset)))[:train_num]), Subset(dataset, list(range(len(dataset)))[-test_num:]) 

train_dataloader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle, num_workers=num_workers)

sde_dyn_learner = SDEGumbelGraphNetwork(in_dim, hidden_dim, delta_t)
sde_net_learner = Gumbel_Generator(node_num, temp, temp_drop_frac)
sde_seq2seq_dyn_learner = DynSeq2Seq(sde_dyn_learner)
sde_seq2seq_net_learner = NetSeq2Seq(sde_net_learner)


ode_dyn_learner = ODEGumbelGraphNetwork(in_dim, hidden_dim, delta_t)
ode_net_learner = Gumbel_Generator(node_num, temp, temp_drop_frac)
ode_seq2seq_dyn_learner = DynSeq2Seq(ode_dyn_learner)
ode_seq2seq_net_learner = NetSeq2Seq(ode_net_learner)


path_check(model_save_path)
path_check(os.path.join(model_save_path, 'base'))
torch.save(sde_seq2seq_dyn_learner, os.path.join(model_save_path, 'base', 'sde_seq2seq_dyn_learner'))
torch.save(sde_seq2seq_net_learner, os.path.join(model_save_path, 'base', 'sde_seq2seq_net_learner'))
torch.save(ode_seq2seq_dyn_learner, os.path.join(model_save_path, 'base', 'ode_seq2seq_dyn_learner'))
torch.save(ode_seq2seq_net_learner, os.path.join(model_save_path, 'base', 'ode_seq2seq_net_learner'))


sde_seq2seq_dyn_learner = sde_seq2seq_dyn_learner.to(device)
sde_seq2seq_net_learner = sde_seq2seq_net_learner.to(device)
ode_seq2seq_dyn_learner = ode_seq2seq_dyn_learner.to(device)
ode_seq2seq_net_learner = ode_seq2seq_net_learner.to(device)


opt_sde_seq2seq_dyn_learner = optim.Adamax(sde_seq2seq_dyn_learner.parameters(), dyn_lr)
opt_sde_seq2seq_net_learner = optim.Adamax(sde_seq2seq_net_learner.parameters(), net_lr)
opt_ode_seq2seq_dyn_learner = optim.Adamax(ode_seq2seq_dyn_learner.parameters(), dyn_lr)
opt_ode_seq2seq_net_learner = optim.Adamax(ode_seq2seq_net_learner.parameters(), net_lr)

for batch_idx, train_data in enumerate(train_dataloader):
    train_data = train_data.to(device)


for epoch_idx in range(epoch_num):
    
    if epoch_idx % iter_step < dyn_step:
        stat = 'dyn'
        sde_mse, sde_mae = train_s_dyn_learner(sde_seq2seq_dyn_learner, sde_seq2seq_net_learner, opt_sde_seq2seq_dyn_learner, train_data, window_size)
        ode_mse, ode_mae = train_s_dyn_learner(ode_seq2seq_dyn_learner, ode_seq2seq_net_learner, opt_ode_seq2seq_dyn_learner, train_data, window_size)
 
    else:
        stat = 'net'
        sde_mse, sde_mae = train_s_net_learner(sde_seq2seq_dyn_learner, sde_seq2seq_net_learner, opt_sde_seq2seq_net_learner, train_data, window_size)
        ode_mse, ode_mae = train_s_net_learner(ode_seq2seq_dyn_learner, ode_seq2seq_net_learner, opt_ode_seq2seq_net_learner, train_data, window_size)
        

        





    print('epoch: {:>2d} training'.format(epoch_idx), end=" ")
    print(stat, end=' ')
    print('sde_mse: {:>8.6f}'.format(sde_mse), end=" ")
    print('sde_mae: {:>8.6f}'.format(sde_mae), end=" ")
    print('ode_mse: {:>8.6f}'.format(ode_mse), end=" ")
    print('ode_mae: {:>8.6f}'.format(ode_mae), end=" ")
    prefect = ('SDE' if sde_mse<=ode_mse else 'ODE')
    print('prefect: ' + '\033[31m{}\033[0m'.format(prefect))





