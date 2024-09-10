import time

import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm


def get_num_samples(sample_arg, dataset_len):
    if sample_arg < 1:
        samples = int(sample_arg * dataset_len)
    else:
        samples = int(min(sample_arg, dataset_len))
    return samples


def train(model, optimizer, train_loader, args, device, emb=None):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    batch_processing_times = []
    for batch_count, data in enumerate(pbar):
        start_time = time.time()
        optimizer.zero_grad()

        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        pi = data.pi if args.use_ph else None
        pe = data.pe if args.use_pe else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, pi, pe)
        loss = get_loss(args.loss)(logits, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        del data
        torch.cuda.empty_cache()
        batch_processing_times.append(time.time() - start_time)

    return total_loss / len(train_loader.dataset)


def auc_loss(logits, y, num_neg=1):
    pos_out = logits[y == 1]
    neg_out = logits[y == 0]
    # hack, should really pair negative and positives in the training set
    if len(neg_out) <= len(pos_out):
        pos_out = pos_out[:len(neg_out)]
    else:
        neg_out = neg_out[:len(pos_out)]
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def bce_loss(logits, y, num_neg=1):
    return BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    else:
        raise NotImplementedError
    return loss
