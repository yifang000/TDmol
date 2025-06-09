from eval_sample import *
import sys
#cuda = 0
cuda = sys.argv[1]
#SET_SAVE = 'test_0322'
SET_SAVE = sys.argv[2]
#SET_SCALE = 5.0
SET_SCALE = float(sys.argv[3])
#SET_EPOCH = 50
SET_EPOCH = int(sys.argv[4])
#SET_ITER = 50
SET_ITER = int(sys.argv[5])

model_path = '/data/yfang/EDM_new/outputs/edm_geom_drugs/'

with open(join(model_path, 'args.pickle'), 'rb') as f:
    args = pickle.load(f)

# CAREFUL with this -->
if not hasattr(args, 'normalization_factor'):
    args.normalization_factor = 1
if not hasattr(args, 'aggregation_method'):
    args.aggregation_method = 'sum'

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(cuda) if args.cuda else "cpu")
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


fix_layer = True
model_local = GCN_guidance_CL(device,fix_layer)
model_local.to(device)
diff_checkpoint_local = torch.load("/data/yfang/EDM_new/model_GCN/model_GCN_1111_fixed_step0_guidance_epoch2000.pt", map_location=device)
model_local.load_state_dict(diff_checkpoint_local, strict=False)
model_local.eval()

max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

from tqdm import tqdm

samples = 1
text = "(S)-etodolac is the S-enantiomer of etodolac. It is a preferential inhibitor of cyclo-oxygenase 2 and a non-steroidal anti-inflammatory, whereas the enantiomer, (R)-etodolac, is inactive. The racemate is commonly used for the treatment of rheumatoid arthritis and osteoarthritis, and for the alleviation of postoperative pain. It has a role as a cyclooxygenase 2 inhibitor and a non-narcotic analgesic. It is an enantiomer of a (R)-etodolac."

#for sample in tqdm(range(samples)):
for sample in range(samples):
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

    x, h, x_clean, h_clean,sim = flow.sample_ob_sim(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=None,
                                  cond_fn=cond_fn_sdg, guidance_kwargs=[text_token,text_mask,model,model_local],
                                  prior_network=None,scale=SET_SCALE,epoch=SET_EPOCH,start_iter=SET_ITER)

    with open('{}_sim.txt'.format(SET_SAVE),'w') as f:
        for line in sim:
            f.write(str(line)+'\n')

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

    with open('{}.txt'.format(SET_SAVE),'a') as f:
        f.write(rdkit_metrics[1][0]+'\n')            
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    molecules['one_hot'].append(one_hot_clean.detach().cpu())
    molecules['x'].append(x_clean.detach().cpu())
    molecules['node_mask'].append(node_mask.detach().cpu())
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

    #np.save('sample_result/mol_clean{}.npy'.format(sample), molecules)

    from qm9.analyze import analyze_stability_for_molecules
    stability_dict_clean, rdkit_metrics_clean = analyze_stability_for_molecules(molecules, dataset_info)
    
    with open('{}_clean.txt'.format(SET_SAVE),'a') as f:
        f.write(rdkit_metrics[1][0]+'\n')

    '''
    mols = [Chem.MolFromSmiles(rdkit_metrics[1][0]),Chem.MolFromSmiles(rdkit_metrics_clean[1][0]),Chem.MolFromSmiles('CCC1=C2C(=CC=C1)C3=C(N2)C(OCC3)(CC)CC(=O)O')]
    fps = [Chem.RDKFingerprint(x) for x in mols]
    sim = DataStructs.FingerprintSimilarity(fps[0], fps[2])
    sim2 = DataStructs.FingerprintSimilarity(fps[1], fps[2])
    print(sim,sim2)
    '''
