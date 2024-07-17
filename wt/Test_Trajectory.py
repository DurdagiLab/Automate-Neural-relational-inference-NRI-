import time
import argparse
import pickle
import os
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from modules import *
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(
    'Neural relational inference for molecular dynamics simulations')
parser.add_argument('--num-residues', type=int, default=406,
                    help='Number of residues of the PDB.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=4,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=6,
                    help='The number of input dimensions used in study( position (X,Y,Z) + velocity (X,Y,Z) ). ')
parser.add_argument('--timesteps', type=int, default=20,
                    help='The number of time steps per sample. Actually is 50')
parser.add_argument('--prediction-steps', type=int, default=1, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units in encoder.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units in decoder.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='rnn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability) in encoder.')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability) in decoder.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=True,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=True,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=True,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--number-expstart', type=int, default=0,
                    help='start number of experiments.')
parser.add_argument('--number-exp', type=int, default=100,
                    help='number of experiments.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
# print all arguments
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# load data
train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_dataset_train_valid_test(
    args.batch_size, args.number_exp, args.number_expstart, args.dims)


# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_residues, args.num_residues]
                   ) - np.eye(args.num_residues)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
    cp=torch.load('logs/encoder_train.pt')
    encoder.load_state_dict(cp)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
    
    cp=torch.load('logs/decoder_train.pt')
    decoder.load_state_dict(cp)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(
        loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder_train.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder_train.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_residues)
tril_indices = get_tril_offdiag_indices(args.num_residues)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

def dis_mat(V):
    Dis=np.zeros((V.size(0),V.size(0)))
    for i in range(V.size(0)):
        for j in range(V.size(0)):
            Dis[i,j]=((V[i,0]-V[j,0])**2+(V[i,1]-V[j,1])**2+(V[i,2]-V[j,2])**2)**(1/2)
    
    return Dis
        

def get_trajec():
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    edges_train = []
    probs_train = []

    encoder.train()
    decoder.train()
    output_tra=[]
    target_tra=[]
    for batch_idx, (data, relations) in enumerate(train_loader):
        
        print(batch_idx)
        if batch_idx==1:
            break
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)
        V=data[0,:,0,:3]
        Dis=dis_mat(V)
        if args.decoder == 'rnn':
            output_tra.append(decoder(data, edges, rel_rec, rel_send, 20,
                             burn_in=True,
                             burn_in_steps=args.timesteps - args.prediction_steps))
        else:
            output_tra.append(decoder(data, edges, rel_rec, rel_send,
                             args.prediction_steps))

        target_tra.append(data[:, :, 1:, :])

        loss_nll = nll_gaussian(output_tra[-1], target_tra[-1], args.var)

    #scheduler.step()
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    output_val=[]
    target_val=[]


    return output_tra,target_tra,output_val,target_val


output_tra,target_tra,output_val,target_val=get_trajec()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)

ax.scatter(target_tra[0][0,:,5,0].cpu(),output_tra[0][0,:,5,0].cpu().detach().numpy())
ax.scatter(target_tra[0][0,:,5,1].cpu(),output_tra[0][0,:,5,1].cpu().detach().numpy())
ax.scatter(target_tra[0][0,:,5,2].cpu(),output_tra[0][0,:,5,2].cpu().detach().numpy())
plt.savefig('logs/fit_scatter.png', dpi=300)
plt.show()


fig2 = plt.figure(figsize=(16, 10))
ax2 = fig2.add_subplot(111)
ax2.plot(target_tra[0][0,:,2,1].cpu())
ax2.plot(output_tra[0][0,:,2,1].cpu().detach().numpy())
plt.savefig('logs/fit_example_y_axis.png', dpi=300)
plt.show()
#plt.close()