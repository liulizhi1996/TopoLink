import torch
from tqdm import tqdm

from src.utils.eval_utils import evaluate_auc, evaluate_hits, evaluate_mrr, evaluate_ogb_rocauc


@torch.no_grad()
def test(model, evaluator, val_loader, test_loader, args, device, emb=None, eval_metric='hits'):
    model.eval()
    pos_val_pred, neg_val_pred, val_pred, val_true = get_preds(model, val_loader, device, args, emb, split='val')
    pos_test_pred, neg_test_pred, test_pred, test_true = get_preds(model, test_loader, device, args, emb, split='test')

    if eval_metric == 'hits':
        results = evaluate_hits(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'mrr':
        results = evaluate_mrr(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)
    elif args.eval_metric == 'rocauc':
        results = evaluate_ogb_rocauc(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    else:
        raise NotImplementedError(f'{eval_metric} is not a valid metric.')

    return results


@torch.no_grad()
def get_preds(model, loader, device, args, emb=None, split=None):
    y_pred, y_true = [], []
    pbar = tqdm(loader, ncols=70)
    for batch_count, data in enumerate(pbar):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        pi = data.pi if args.use_ph else None
        pe = data.pe if args.use_pe else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, pi, pe)
        y_true.append(data.y.view(-1).cpu().to(torch.float))
        y_pred.append(logits.view(-1).cpu())
        del data
        torch.cuda.empty_cache()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = pred[true == 1]
    neg_pred = pred[true == 0]
    return pos_pred, neg_pred, pred, true


def get_split_samples(split, args, dataset_len):
    samples = dataset_len
    if split == 'train':
        if args.dynamic_train:
            samples = get_num_samples(args.train_samples, dataset_len)
    elif split in {'val', 'valid'}:
        if args.dynamic_val:
            samples = get_num_samples(args.val_samples, dataset_len)
    elif split == 'test':
        if args.dynamic_test:
            samples = get_num_samples(args.test_samples, dataset_len)
    else:
        raise NotImplementedError(f'split: {split} is not a valid split')
    return samples


def get_num_samples(sample_arg, dataset_len):
    if sample_arg < 1:
        samples = int(sample_arg * dataset_len)
    else:
        samples = int(min(sample_arg, dataset_len))
    return samples
