import argparse
from math import inf

import torch
from ogb.linkproppred import Evaluator

from src.utils.data_utils import get_data, get_loaders
from src.utils.misc_utils import set_seed, str2bool
from src.models.model import TopoLink
from src.train import train
from src.inference import test

# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def select_embedding(args, num_nodes, device):
    if args.train_node_embedding:
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None
    return emb


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    dataset, splits, directed, eval_metric = get_data(args)
    train_loader, val_loader, test_loader = get_loaders(args, dataset, splits)
    if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
        evaluator = Evaluator(name=args.dataset_name)
    elif args.eval_metric == 'mrr':
        evaluator = Evaluator(name='ogbl-citation2')  # this sets MRR as the metric
    elif args.eval_metric == 'auc':
        evaluator = Evaluator(name='ogbl-vessel')  # this sets AUC as the metric
    else:
        evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
    emb = select_embedding(args, dataset.data.num_nodes, device)

    model = TopoLink(args.hidden_channels, args.num_seal_layers, args.max_z, args.sortpool_k, dataset,
                     args.dynamic_train, use_feature=args.use_feature, use_node_label=args.use_node_label,
                     node_embedding=emb, gnn_dropout=args.gnn_dropout,
                     attn_dropout=args.attn_dropout, mlp_dropout=args.mlp_dropout,
                     attn_type=args.attn_type, num_heads=args.num_heads,
                     use_ph=args.use_ph, pi_dim=args.pi_dim, patch_size=args.patch_size,
                     vit_dim=args.vit_dim, vit_depth=args.vit_depth,
                     vit_out_dim=args.vit_out_dim, vit_headers=args.vit_headers,
                     vit_mlp_dim=args.vit_mlp_dim, use_pe=args.use_pe,
                     walk_length=args.walk_length, out_mlp_dim=args.out_mlp_dim).to(device)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.AdamW(params=parameters, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train(model, optimizer, train_loader, args, device)
        if (epoch + 1) % args.eval_steps == 0:
            results = test(model, evaluator, val_loader, test_loader, args, device,
                           eval_metric=eval_metric)
            for key, result in results.items():
                val_res, test_res = result
                to_print = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ' \
                           f'Valid: {100 * val_res:.2f}%, Test: {100 * test_res:.2f}%'
                print(key)
                print(to_print)


def parse_args():
    # Data settings
    parser = argparse.ArgumentParser(description='TopoLink')
    parser.add_argument('--dataset_name', type=str, default='Cora')
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not '
                             'appear in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    parser.add_argument('--seed', type=int, default=23, help='seed for reproducibility')
    # Model settings
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--num_seal_layers', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0)
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--mlp_dropout', type=float, default=0)
    parser.add_argument('--attn_type', type=str, default='sga')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--out_mlp_dim', type=int, default=128)
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    parser.add_argument('--use_pe', type=str2bool, default=False,
                        help="whether to add positional encoding in GNN")
    parser.add_argument('--use_node_label', type=str2bool, default=True,
                        help="whether to add node labeling in GNN")
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl')
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--walk_length', type=int, default=32,
                        help='The number of random walk steps in random walk positional encoding')
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Persistent homology settings
    parser.add_argument('--use_ph', type=str2bool, default=True, help='whether to consider persistence homology')
    parser.add_argument('--extend', type=str2bool, default=False, help='whether to use extend persistence')
    parser.add_argument('--pi_dim', type=int, default=50, help='resolution of persistent image')
    parser.add_argument('--filtration', type=str, default='degree',
                        choices=['degree', 'betweenness', 'communicability', 'eigenvector', 'closeness', 'ricci'])
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--vit_dim', type=int, default=128)
    parser.add_argument('--vit_depth', type=int, default=2)
    parser.add_argument('--vit_out_dim', type=int, default=256)
    parser.add_argument('--vit_headers', type=int, default=4)
    parser.add_argument('--vit_mlp_dim', type=int, default=128)
    # Testing settings
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='hits',
                        choices=('hits', 'mrr', 'auc', 'rocauc'))
    parser.add_argument('--K', type=int, default=100, help='the hit rate @K')

    args = parser.parse_args()
    if args.dataset_name == 'ogbl-ddi':
        args.use_feature = 0  # dataset has no features
    return args


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
