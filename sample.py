from eval_sample import *

model_path = '/data/yfang/EDM_new/outputs/edm_geom_drugs/'

with open(join(model_path, 'args.pickle'), 'rb') as f:
    args = pickle.load(f)

# CAREFUL with this -->
if not hasattr(args, 'normalization_factor'):
    args.normalization_factor = 1
if not hasattr(args, 'aggregation_method'):
    args.aggregation_method = 'sum'

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device
dtype = torch.float32
print(args)

dataset_info = get_dataset_info(args.dataset, args.remove_h)
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

flow, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
flow.to(device)

fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'

flow_state_dict = torch.load(join(model_path, fn),map_location=device)

flow.load_state_dict(flow_state_dict)

###conditional sampling
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem

##loading CL model (full step model)
from finetune_CL_new import *
diff_checkpoint = torch.load("/data/yfang/EDM_new/model_GCN/model_GCN_1111_fixed_full200_epoch20_guidance_epoch26.pt", map_location=device)
model.load_state_dict(diff_checkpoint, strict=False)
model.eval()

max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

from tqdm import tqdm

def generate_mol(text,samples,index,save_path,scale=5.0):
    for sample in tqdm(range(samples)):

        n_samples=1
        batch_size = 1
        nodesxsample = nodes_dist.sample(n_samples)

        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)

        context = None

        ##text tokenizer
        text_input = gd.text_tokenizer(text, truncation=True, padding = 'max_length', 
                                       max_length=text_trunc_length, return_tensors = 'pt')

        text_token = text_input['input_ids']
        text_mask = text_input['attention_mask']
        text_token = text_token.to(device)
        text_mask = text_mask.to(device)

        x, h, x_clean, h_clean = flow.sample_ob(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=None,
                                      cond_fn=cond_fn_sdg, guidance_kwargs=[text_token,text_mask,model],prior_network=None,scale=scale)

        assert_correctly_masked(x, node_mask)
        assert_correctly_masked(x_clean, node_mask)
        #assert_mean_zero_with_mask(x, node_mask)
        one_hot = h['categorical']
        charges = h['integer']
        one_hot_clean = h_clean['categorical']
        charges_clean = h_clean['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        assert_correctly_masked(one_hot_clean.float(), node_mask)
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        
        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        from qm9.analyze import analyze_stability_for_molecules
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info)
        
        np.save('{}/mol{}.npy'.format(save_path,sample), molecules)
        try:
            with open('{}/mol_{}.txt'.format(save_path,index),'a') as f:
                f.write(rdkit_metrics[1][0]+'\n')   
        except:
            with open('{}/mol_{}.txt'.format(save_path,index),'a') as f:
                f.write('\n')              
        
        molecules_clean = {'one_hot': [], 'x': [], 'node_mask': []}
        molecules_clean['one_hot'].append(one_hot_clean.detach().cpu())
        molecules_clean['x'].append(x_clean.detach().cpu())
        molecules_clean['node_mask'].append(node_mask.detach().cpu())
        molecules_clean = {key: torch.cat(molecules_clean[key], dim=0) for key in molecules_clean}
        
        np.save('{}/mol_clean{}.npy'.format(save_path,sample), molecules_clean)
        
        stability_dict_clean, rdkit_metrics_clean = analyze_stability_for_molecules(molecules_clean, dataset_info)
        
        try:
            with open('{}/mol_clean_{}.txt'.format(save_path,index),'a') as f:
                f.write(rdkit_metrics_clean[1][0]+'\n')
        except:
            with open('{}/mol_clean_{}.txt'.format(save_path,index),'a') as f:
                f.write('\n')            
        '''
        
        mols = [Chem.MolFromSmiles(rdkit_metrics[1][0]),Chem.MolFromSmiles(rdkit_metrics_clean[1][0]),Chem.MolFromSmiles('CCC1=C2C(=CC=C1)C3=C(N2)C(OCC3)(CC)CC(=O)O')]
        fps = [Chem.RDKFingerprint(x) for x in mols]
        sim = DataStructs.FingerprintSimilarity(fps[0], fps[2])
        sim2 = DataStructs.FingerprintSimilarity(fps[1], fps[2])
        print(sim,sim2)
        '''

            
if __name__ == '__main__':
    import os
    samples = 100
    save_path = 'sample_result_test_scale5'
    with open('/data/yfang/txt2mol/data/test.txt') as f:
        description = f.readlines()
    text_description = [i.split('\t')[-1].strip('\n') for i in description]
    for text in text_description:
        index = text_description.index(text)
        generate_mol(text,samples,index,save_path)
        
        data_list = []
        data_list_clean = []
        for i in range(samples):
            if os.path.exists('{}/mol{}.npy'.format(save_path,i)):
                data = np.load('{}/mol{}.npy'.format(save_path,i),allow_pickle=True)
            else:
                data = np.array({'one_hot':'null','x':'null','node_mask':'null'})
            if os.path.exists('{}/mol_clean{}.npy'.format(save_path,i)):
                data_clean = np.load('{}/mol_clean{}.npy'.format(save_path,i),allow_pickle=True)
            else:
                data_clean = np.array({'one_hot':'null','x':'null','node_mask':'null'})
            data_list.append(data)
            data_list_clean.append(data_clean)
            os.remove('{}/mol{}.npy'.format(save_path,i))
            os.remove('{}/mol_clean{}.npy'.format(save_path,i))
            
        np.save('{}/mol_{}.npy'.format(save_path,index), data_list)
        np.save('{}/mol_clean_{}.npy'.format(save_path,index), data_list_clean)
