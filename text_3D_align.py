import os
import os.path as osp
import shutil
import time
import csv
import math
import pickle
import numpy as np
import sys
sys.path.append('/data/yfang/txt2mol/code')
from models import MLPModel,GCNModel
from dataloaders import get_dataloader, GenerateData, get_graph_data, get_attention_graph_data, GenerateDataAttention, get_attention_dataloader
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch_geometric
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torch_geometric
from tqdm import tqdm

CHECKPOINT = '/data/yfang/txt2mol/result_graphfinal_weights.40.pt'

device = torch.device("cuda:2")

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def get_txt_emb(cid,dataset_label):
    if dataset_label == 'train':
        emb_file = np.load('/data/yfang/txt2mol/result_graph/embeddings/chem_embeddings_train.npy')
        cid_file = np.load('/data/yfang/txt2mol/result_graph/embeddings/cids_train.npy')
    elif dataset_label == 'val':
        emb_file = np.load('/data/yfang/txt2mol/result_graph/embeddings/chem_embeddings_val.npy')
        cid_file = np.load('/data/yfang/txt2mol/result_graph/embeddings/cids_val.npy')
    else:
        return 0
    index = [list(cid_file).index(i) for i in cid]
    emb = [emb_file[j] for j in index]
    return torch.tensor(emb)


class GCN_guidance_CL_Oldversion(nn.Module):
    def __init__(self,device,fix_layer=True):
        super().__init__()
        self.device = device
        self.fix_layer = fix_layer
        #self.preprocess = transforms.Normalize(mean=(x,y,z), std=(x1,y1,z1)) #centralization
        GCN_model = GCNModel(num_node_features=300, ninp = 768, nhid = 600, nout = 300, graph_hidden_channels = 600)
        GCN_model.load_state_dict(torch.load(CHECKPOINT,map_location=self.device))
        
        # Mol Encoder
        #GCN layers - Readout - MLP layers
        self.mol_GCN = nn.Sequential(
            GCN_model.conv1,
            GCN_model.relu,
            GCN_model.conv2,
            GCN_model.relu,
            GCN_model.conv3,
        )
        
        self.mol_frozen = nn.Sequential(
            GCN_model.mol_hidden1,
            GCN_model.relu,
            GCN_model.mol_hidden2,
            GCN_model.relu,
        )
        self.mol_hidden = GCN_model.mol_hidden3
        self.mol_ln = GCN_model.ln1
        
        #Text Encoder
        self.text_transformer = GCN_model.text_transformer_model
        self.text_hidden = GCN_model.text_hidden1
        self.text_ln = GCN_model.ln2

        self.after_load()
        self.define_finetune()
        
        self.logit_scale = nn.Parameter(torch.Tensor([0.07]))
        #map noised graph feature dim (5) to 300
        self.transform_nn = nn.Linear(19,300)
        
    def after_load(self):
        #增加time-embedding输入
        #time原始维度（batch_size,）
        #经过timestep_embedding函数正弦编码为（batch_size,128）
        #再经过self.time_embed变为（batch_size,512）
        #再经过emb_layers 变为（batch_size,sum） 分配到mol encoder的每一层
        #一共三层，需要计算6次,因为有（scale,shift）两次变换，平移和放缩， 维度的计算也是跟mol encoder的每一层保持一致
        
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        emb_dim = [600, 600, 300]
        self.emb_layers = nn.Sequential(nn.ReLU(), nn.Linear(512, sum(emb_dim) * 2))
        self.split_idx = [600, 1200, 1800, 2400, 2700, 3000]

    def define_finetune(self):
        self.train()
        # freeze mol encoder
        if self.fix_layer:
            for param in self.mol_GCN.parameters():
                param.requires_grad = False
            for param in self.mol_frozen.parameters():
                param.requires_grad = False
            for param in self.mol_hidden.parameters():
                param.requires_grad = False
            self.time_embed.requires_grad = True
            self.emb_layers.requires_grad = True
                
    def encode_mol_raw(self,mol_batch):
        x = mol_batch.x
        edge_index = mol_batch.edge_index
        batch = mol_batch.batch
        
        x = self.mol_GCN[1](self.mol_GCN[0](x, edge_index))
        x = self.mol_GCN[3](self.mol_GCN[2](x, edge_index))
        x = self.mol_GCN[4](x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.mol_frozen[1](self.mol_frozen[0](x))
        x = self.mol_frozen[3](self.mol_frozen[2](x))
        x = self.mol_hidden(x)
        x = self.mol_ln(x)
        return x
        

    def encode_mol(self, mol_batch_x, mol_batch_edge_index, mol_batch_batch, t, node_mask=None):
        #mol = self.preprocess(mol)
        emb = self.time_embed(timestep_embedding(t, 128))
        emb = emb.squeeze(dim=1)
        emb_out = torch.tensor_split(self.emb_layers(emb), self.split_idx, dim=1)
        
        x = mol_batch_x
        edge_index = mol_batch_edge_index
        batch = mol_batch_batch
        #print(x.shape)
        
        x = self.transform_nn(x)    #这个linear层会有bias，改变了node_mask
        node_mask_ = node_mask.reshape(node_mask.shape[0]*node_mask.shape[1],1)
        x = x*node_mask_
        x = x.squeeze()  #从[1,181,300]到[181,300]
        
        x = self.mol_GCN[1](self.mol_GCN[0](x, edge_index))
        x = self.mol_GCN[3](self.mol_GCN[2](x, edge_index))
        x = self.mol_GCN[4](x, edge_index)
        x = global_mean_pool(x, batch)
        
        #layer1
        x = self.mol_frozen[1](self.mol_frozen[0](x))
        x = x * (1 + emb_out[0]) + emb_out[1]
        #layer2
        x = self.mol_frozen[3](self.mol_frozen[2](x))
        x = x * (1 + emb_out[2]) + emb_out[3]
        #layer3
        x = self.mol_hidden(x)
        x = x * (1 + emb_out[4]) + emb_out[5]
        #layer norm
        
        x = self.mol_ln(x)
        #print(x.shape)

        return x

    def encode_text(self, text, text_mask=None):
        text_encoder_output = self.text_transformer(text,attention_mask=text_mask)
        x = text_encoder_output['pooler_output']
        x = self.text_hidden(x)
        x = self.text_ln(x)
        return x


    def forward(self, mol_batch_x, mol_batch_edge_index, mol_batch_batch ,text_features, timesteps, node_mask):
        
        mol_features = self.encode_mol(mol_batch_x, mol_batch_edge_index, mol_batch_batch, timesteps, node_mask)
        #因为这里fine tuning的时候是把原mol特征和noised_mol特征，所以这里不需要text_features
        #（而且调用的时候text_features指代的是原mol特征，而不是文本特征）
        #text_features = self.encode_text(text)
        return mol_features, text_features

    def training_step(self, batch, batch_idx):
        mol_batch, text = batch
        bs = mol_batch.batch.size(0)

        mol_features, text_features = self(mol_batch, text)

        # normalized features
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_mol = logit_scale * mol_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ mol_features.t()

        label = torch.arange(bs).long()
        label = label.to(mol_batch.device)

        loss_i = F.cross_entropy(logits_per_mol, label)
        loss_t = F.cross_entropy(logits_per_text, label)

        loss = (loss_i + loss_t) / 2

        return loss


class GCN_guidance_CL(nn.Module):
    def __init__(self,device,fix_layer=True):
        super().__init__()
        self.device = device
        self.fix_layer = fix_layer
        #self.preprocess = transforms.Normalize(mean=(x,y,z), std=(x1,y1,z1)) #centralization
        GCN_model = GCNModel(num_node_features=300, ninp = 768, nhid = 600, nout = 300, graph_hidden_channels = 600)
        GCN_model.load_state_dict(torch.load(CHECKPOINT,map_location=self.device))
        
        # Mol Encoder
        #GCN layers - Readout - MLP layers
        self.mol_GCN = nn.Sequential(
            GCN_model.conv1,
            GCN_model.relu,
            GCN_model.conv2,
            GCN_model.relu,
            GCN_model.conv3,
        )
        
        self.mol_frozen = nn.Sequential(
            GCN_model.mol_hidden1,
            GCN_model.relu,
            GCN_model.mol_hidden2,
            GCN_model.relu,
        )
        self.mol_hidden = GCN_model.mol_hidden3
        self.mol_ln = GCN_model.ln1
        
        # A copy of raw mol encoder
        self.mol_GCN_new = nn.Sequential(
            GCN_model.conv1,
            GCN_model.relu,
            GCN_model.conv2,
            GCN_model.relu,
            GCN_model.conv3,
        )
        
        self.mol_frozen_new = nn.Sequential(
            GCN_model.mol_hidden1,
            GCN_model.relu,
            GCN_model.mol_hidden2,
            GCN_model.relu,
        )
        self.mol_hidden_new = GCN_model.mol_hidden3
        self.mol_ln_new = GCN_model.ln1  
        
        
        #Text Encoder
        self.text_transformer = GCN_model.text_transformer_model
        self.text_hidden = GCN_model.text_hidden1
        self.text_ln = GCN_model.ln2

        self.after_load()
        self.define_finetune()
        
        self.logit_scale = nn.Parameter(torch.Tensor([0.07]))
        #map noised graph feature dim (5) to 300
        self.transform_nn = nn.Linear(19,300)
        
    def after_load(self):
        #增加time-embedding输入
        #time原始维度（batch_size,）
        #经过timestep_embedding函数正弦编码为（batch_size,128）
        #再经过self.time_embed变为（batch_size,512）
        #再经过emb_layers 变为（batch_size,sum） 分配到mol encoder的每一层
        #一共三层，需要计算6次,因为有（scale,shift）两次变换，平移和放缩， 维度的计算也是跟mol encoder的每一层保持一致
        
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        emb_dim = [600, 600, 300]
        self.emb_layers = nn.Sequential(nn.ReLU(), nn.Linear(512, sum(emb_dim) * 2))
        self.split_idx = [600, 1200, 1800, 2400, 2700, 3000]

    def define_finetune(self):
        self.train()
        # freeze mol encoder
        if self.fix_layer:
            for param in self.mol_GCN.parameters():
                param.requires_grad = False
            for param in self.mol_frozen.parameters():
                param.requires_grad = False
            for param in self.mol_hidden.parameters():
                param.requires_grad = False
            for param in self.mol_ln.parameters():
                param.requires_grad = False
            #self.time_embed.requires_grad = True
            #self.emb_layers.requires_grad = True
                
    def encode_mol_raw(self,mol_batch):
        x = mol_batch.x
        edge_index = mol_batch.edge_index
        batch = mol_batch.batch
        
        x = self.mol_GCN[1](self.mol_GCN[0](x, edge_index))
        x = self.mol_GCN[3](self.mol_GCN[2](x, edge_index))
        x = self.mol_GCN[4](x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.mol_frozen[1](self.mol_frozen[0](x))
        x = self.mol_frozen[3](self.mol_frozen[2](x))
        x = self.mol_hidden(x)
        x = self.mol_ln(x)
        return x
        

    def encode_mol(self, mol_batch_x, mol_batch_edge_index, mol_batch_batch, t, node_mask=None):
        #mol = self.preprocess(mol)
        emb = self.time_embed(timestep_embedding(t, 128))
        emb = emb.squeeze(dim=1)
        emb_out = torch.tensor_split(self.emb_layers(emb), self.split_idx, dim=1)
        
        x = mol_batch_x
        edge_index = mol_batch_edge_index
        batch = mol_batch_batch
        #print(x.shape)
        
        x = self.transform_nn(x)    #这个linear层会有bias，改变了node_mask
        node_mask_ = node_mask.reshape(node_mask.shape[0]*node_mask.shape[1],1)
        x = x*node_mask_
        x = x.squeeze()  #从[1,181,300]到[181,300]
        
        x = self.mol_GCN_new[1](self.mol_GCN_new[0](x, edge_index))
        x = self.mol_GCN_new[3](self.mol_GCN_new[2](x, edge_index))
        x = self.mol_GCN_new[4](x, edge_index)
        x = global_mean_pool(x, batch)
        
        #layer1
        x = self.mol_frozen_new[1](self.mol_frozen_new[0](x))
        x = x * (1 + emb_out[0]) + emb_out[1]
        #layer2
        x = self.mol_frozen_new[3](self.mol_frozen_new[2](x))
        x = x * (1 + emb_out[2]) + emb_out[3]
        #layer3
        x = self.mol_hidden_new(x)
        x = x * (1 + emb_out[4]) + emb_out[5]
        #layer norm
        
        x = self.mol_ln_new(x)
        #print(x.shape)

        return x

    def encode_text(self, text, text_mask=None):
        text_encoder_output = self.text_transformer(text,attention_mask=text_mask)
        x = text_encoder_output['pooler_output']
        x = self.text_hidden(x)
        x = self.text_ln(x)
        return x


    def forward(self, mol_batch_x, mol_batch_edge_index, mol_batch_batch ,text_features, timesteps, node_mask):
        
        mol_features = self.encode_mol(mol_batch_x, mol_batch_edge_index, mol_batch_batch, timesteps, node_mask)
        #因为这里fine tuning的时候是把原mol特征和noised_mol特征，所以这里不需要text_features
        #（而且调用的时候text_features指代的是原mol特征，而不是文本特征）
        #text_features = self.encode_text(text)
        return mol_features, text_features

    def training_step(self, batch, batch_idx):
        mol_batch, text = batch
        bs = mol_batch.batch.size(0)

        mol_features, text_features = self(mol_batch, text)

        # normalized features
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_mol = logit_scale * mol_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ mol_features.t()

        label = torch.arange(bs).long()
        label = label.to(mol_batch.device)

        loss_i = F.cross_entropy(logits_per_mol, label)
        loss_t = F.cross_entropy(logits_per_text, label)

        loss = (loss_i + loss_t) / 2

        return loss
        
#fix_layer = False
fix_layer = True
model = GCN_guidance_CL(device,fix_layer)
model.to(device)


data_path = '/data/yfang/txt2mol/data/'
path_token_embs = osp.join(data_path, "token_embedding_dict.npy")
path_train = osp.join(data_path, "training.txt")
path_val = osp.join(data_path, "val.txt")
path_test = osp.join(data_path, "test.txt")
path_molecules = osp.join(data_path, "ChEBI_defintions_substructure_corpus.cp")
graph_data_path = osp.join(data_path, "mol_graphs.zip")

text_trunc_length = 256
batch_size=256
params = {'batch_size': batch_size,'num_workers': 1}
gd = GenerateData(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)
training_generator, validation_generator, test_generator = get_dataloader(gd, params)
graph_batcher_tr, graph_batcher_val, graph_batcher_test = get_graph_data(gd, graph_data_path)

from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0005)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.9)


def get_noised_data(data_label='training',data_dir='/data/yfang/GeoLDM-main/GeoLDM-main/noised_data_EDM_200full_new',time_label=0):
    data = np.load('{}/{}/data{}.npy'.format(data_dir,data_label,time_label),allow_pickle=True)
    graph = data.item()['graph']
    noised_time = data.item()['time']
    cid = data.item()['cid']
    node_mask = data.item()['node_mask']
    num = len(cid)
    for i in range(num):
        graph[i].y = [noised_time[i],cid[i],node_mask[i]]
    return graph,num
   

def train(date, epoch, device,is_training=True, use_fc_graph = True):
    #for time_label in range(1):
    for time_label in range(10):
        if is_training:
            model.train()
            dataset_label = 'train'
            graph,num = get_noised_data(data_label='training',data_dir='/data/yfang/GeoLDM-main/GeoLDM-main/noised_data_EDM_200full_new',time_label=time_label)
            #graph_batch_2D = graph_batcher_tr
        else:
            model.eval()
            dataset_label = 'val'
            graph,num = get_noised_data(data_label='validation',data_dir='/data/yfang/GeoLDM-main/GeoLDM-main/noised_data_EDM_200full_new',time_label=time_label)
            #graph_batch_2D = graph_batcher_val
        
        print('loading {} mol graphs from {} of {},'.format(num,dataset_label,time_label))
            
        dataloader = torch_geometric.loader.DataLoader(dataset=graph,batch_size=256,shuffle=False)
            
        for _,batch_noise_graph in enumerate(dataloader):
            
            batch_noise_graph = batch_noise_graph.to(device)
            batch_noised_time = batch_noise_graph.y[0]
            batch_cid = batch_noise_graph.y[1]
            node_mask = batch_noise_graph.y[2]
            batch_noised_time = torch.tensor(batch_noised_time).to(device)
            node_mask = torch.tensor(node_mask).to(device)
            
            #graph_batch = graph_batch_2D(batch_cid)
            #graph_batch = graph_batch.to(device)
            #text embedding
            text_emb = get_txt_emb(batch_cid,dataset_label)
            text_emb = text_emb.to(device)
            
            ground_truth = torch.arange(len(batch_cid), dtype=torch.long)
            ground_truth = ground_truth.to(device)
            
            ##2D embedding
            #mol_features_raw = model.encode_mol_raw(graph_batch)
            mol_features_raw = text_emb
            
            #3D embedding
            mol_batch_x = batch_noise_graph.x
            mol_batch_edge_index = batch_noise_graph.edge_index
            mol_batch_batch = batch_noise_graph.batch

            mol_features, mol_features_raw = model(mol_batch_x, mol_batch_edge_index,mol_batch_batch, mol_features_raw, batch_noised_time, node_mask)
            
            #compute loss
            mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
            mol_features_raw = mol_features_raw / mol_features_raw.norm(dim=-1, keepdim=True)
            losses = {}
            logits_per_mol = 100.0 * mol_features @ mol_features_raw.t()
            logits_per_mol_raw = logits_per_mol.t()

            loss_i2t = F.cross_entropy(logits_per_mol, ground_truth, reduction='none')
            loss_t2i = F.cross_entropy(logits_per_mol_raw, ground_truth, reduction='none')
            
            loss = loss_i2t + loss_t2i
            #loss = loss_t2i

            losses[f"train_loss_i2t"] = loss_i2t.detach()
            losses[f"train_loss_t2i"] = loss_t2i.detach()
            losses[f"train_loss"] = loss.detach()
            log(date,dataset_label,losses,epoch)
            del losses

            loss = loss.mean()
            log2(date,dataset_label,loss,epoch)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

#从sdf提取3d graph
def process_sdf(dataset,cid,path='/data/yfang/txt2mol/data/3Dstruc/'):
    with open('{}{}/{}.sdf'.format(path,dataset,cid)) as f:
        lf = f.readlines()
    l_atom_num, l_bond_num = lf[3][0:3],lf[3][3:6] #number_of_atoms, number_of_bonds
    l_atom_num, l_bond_num = int(l_atom_num),int(l_bond_num)
    l_atom = lf[4:4+l_atom_num] #lines contain atom information
    l_atom_type = [i[30:32] for i in l_atom]
    l_cord = [[i[0:10],i[10:20],i[20:30]] for i in l_atom]
    l_cord = np.array(l_cord,dtype='float64')
    
    l_bond =  lf[4+l_atom_num:4+l_atom_num+l_bond_num] #lines contain bond information
    l_bond = [[i[:3],i[3:6],i[6:9]] for i in l_bond]
    l_bond = np.array(l_bond,dtype='int') 
    #print(l_bond.shape) #(l_bond_num,3)
    return l_cord,l_bond,l_atom_type

#将原子类型转为one-hot，根据diffusion模型中的dataset_info['atom_encoder']
def atomtype2onehot(atom_type,atom_encoder={'H': 0,'B': 1,'C': 2,'N': 3,'O': 4,'F': 5,'Al': 6,'Si': 7,'P': 8,'S': 9,'Cl': 10,'As': 11,'Br': 12,'I': 13,'Hg': 14,'Bi': 15}):
    num = atom_encoder[atom_type]
    one_hot = torch.zeros(16)
    one_hot[num] = 1
    return one_hot

#将原子补0到相同原子数，按理说是到181个原子（跟node mask一致）
def padding_atom(atom,padding_size = 181):
    padding  = torch.zeros(padding_size,atom.shape[1])
    padding[:atom.shape[0], :] = atom
    return padding
    
def get_batch_x_one_hot(dataset,batch,get_edge_mask=False):
    batch_cids = batch[0]['molecule']['cid']
    tmp_x = []
    tmp_one_hot = []
    node_mask = torch.zeros(len(batch_cids),181,1)

    if get_edge_mask:
        tmp_edge_mask = torch.zeros(len(batch_cids),181,181,1)

    for cid in batch_cids:
        batch_index = batch_cids.index(cid)
        l_cord,l_bond,l_atom_type = process_sdf(dataset,cid)
        l_one_hot = [atomtype2onehot(atom.strip(' ')).unsqueeze(dim=0) for atom in l_atom_type]
        one_hot = torch.cat(l_one_hot)
        one_hot = padding_atom(one_hot).unsqueeze(dim=0)
        l_cord = torch.tensor(l_cord)
        x = padding_atom(l_cord).unsqueeze(dim=0)
        tmp_x.append(x)
        tmp_one_hot.append(one_hot)

        for i in range(l_cord.shape[0]):
            node_mask[batch_index][i] = 1

        if get_edge_mask:
            for bond in l_bond:
                tmp_edge_mask[batch_index][bond[0]-1][bond[1]-1] = 1
    
    if get_edge_mask:
        edge_mask = tmp_edge_mask.view(len(batch_cids) * 181 * 181, 1)
    else:
        edge_mask = None

    batch_x = torch.cat(tmp_x,dim = 0)
    batch_one_hot = torch.cat(tmp_one_hot,dim = 0)

    return batch_x,batch_one_hot,node_mask,edge_mask

#batch_x   torch.Size([batch_size, 181, 3])
#batch_one_hot torch.Size([batch_size, 181, 16])

def get_fc_edge_from_node(node_mask):
    b,n,_ = node_mask.shape
    edge_mask = torch.zeros(b,n,n,1)
    
    for b_ in range(b):
        for n_ in range(n):
            if node_mask[b_][n_]==1:
                for n_2 in range(n):
                    if n_2 != n_:
                        edge_mask[b_][n_][n_2] = 1
    edge_mask = edge_mask.view(b*n*n, 1)
    return edge_mask


def data2fc_graph(mol_data,node_mask,batch_cids):
    graph_list=[]
    for mol_idx in range(len(batch_cids)):
        mol_feature = mol_data[mol_idx]
        node_mask_ = node_mask[mol_idx]
        num = int(torch.sum(node_mask_))
        ##两两成边（除了自己）
        bond_start_idx = [i for i in range(num) for _ in range(num-1)]
        bond_end_idx = [i for j in range(num) for i in range(num) if i != j]

        edge_index = torch.tensor([bond_start_idx,bond_end_idx],dtype = torch.long)        
        graph = Data(x=mol_feature, edge_index=edge_index)
        graph_list.append(graph)
    return graph_list

def data2graph(mol_data,batch_cids,edge_type=['sdf','distance'][0],edge_weight=[0,1][0]):
    graph_list=[]
    for mol_idx in range(len(batch_cids)):
        mol_feature = mol_data[mol_idx]
        
        #获取边（根据sdf）
        if edge_type=='sdf':
            _,l_bond,_ = process_sdf('train',batch_cids[mol_idx])
        #获取边（根据距离）
        elif edge_type=='distance':
            pass
        
        bond_start_idx = l_bond[:,0]-1
        bond_end_idx = l_bond[:,1]-1
        #无向图的边需要*2
        bond_start_idx_re = np.hstack((bond_start_idx,bond_end_idx))
        bond_end_idx_re = np.hstack((bond_end_idx,bond_start_idx))
        
        #边的index和特征
        edge_index = torch.tensor([bond_start_idx_re,bond_end_idx_re],dtype = torch.long)
        
        edge_attr = None
        if edge_weight:
            bond_weight = l_bond[:,2]
            bond_weight_re = np.hstack((bond_weight,bond_weight))
            edge_attr = torch.tensor(bond_weight_re,dtype = torch.float)
            
        graph = Data(x=mol_feature, edge_index=edge_index, edge_attr=edge_attr)
        graph_list.append(graph)
    return graph_list


def text_loss(source, target):
    source_feat = source[-1] / source[-1].norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    return (source_feat * target).sum(1)


#add node_mask to mask gradient
def cond_fn_sdg(model, x, node_mask, t, text, scale=10.0, text_mask=None, prior_network=None):
    with torch.no_grad():
        text_features = model.encode_text(text,text_mask)
        if prior_network:
            text_features = prior_network(text_features)
            text_features = text_features.squeeze(-1)  #from (1,300,1) to (1,300)
    with torch.enable_grad():
        x_ = x.x.detach().requires_grad_(True)
        edge_index_ = x.edge_index.detach()
        batch_ = x.batch.detach()
        #mol_features = model.encode_mol_list(x_in, t)
        #这里原先是输出一个list的特征，对应模型的不同层输出的特征
        x_ = x_.to(device)
        edge_index_ = edge_index_.to(device)
        batch_ = batch_.to(device)
        
        mol_features = model.encode_mol(x_,edge_index_,batch_, t, node_mask)
        #print('mol feat {}'.format(mol_features.shape))
        #print('text feat {}'.format(text_features.shape))
        #如果只有一个text guide,那text features维度应该是(1,300)
        #而mol features是(batchsize,300)
        #应该把text features扩维到(batchsize,300)
        text_features = text_features.repeat(mol_features.size(0), 1)
        loss_text = text_loss(mol_features, text_features)
        #print('loss_text {}'.format(loss_text.shape))
        
        grad = torch.autograd.grad(loss_text.sum(), x_)[0]
        grad = grad.reshape(1,181,19)
        grad = grad * node_mask
        grad = grad * scale
        #print(grad.shape)
        return grad

def log(date,dataset,losses,epoch):
    with open('loss_GCN/{}_{}_losses.txt'.format(date,dataset),'a') as f:
        f.write('train_loss_i2t_epoch{}: {}\n'.format(epoch,losses[f"train_loss_i2t"]))
        f.write('train_loss_t2i_epoch{}: {}\n'.format(epoch,losses[f"train_loss_t2i"]))
        f.write('train_loss_epoch{}: {}\n'.format(epoch,losses[f"train_loss"]))
        
def log2(date,dataset,loss,epoch):        
    with open('loss_GCN/{}_{}_losses_mean.txt'.format(date,dataset),'a') as f:
        f.write('train_loss_mean_epoch{}: {}\n'.format(epoch,loss))

if __name__ == '__main__':        
    for epoch in tqdm(range(50)):
        #date = '1111_fixed_step0'
        date = '0306_fixed_10'
        train(date,epoch,device=device,is_training=True,use_fc_graph=True)
        train(date,epoch,device=device,is_training=False,use_fc_graph=True) #eval
        if (epoch+1)%2==0:
            torch.save(model.state_dict(), 'model_GCN/model_GCN_{}_guidance_epoch{}.pt'.format(date,epoch+1))            

