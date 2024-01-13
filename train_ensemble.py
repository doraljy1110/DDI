import torch
from torch import optim
from torch.utils.data import DataLoader
import encoder.AMDE_implementations
from losses import LOSS_FUNCTIONS
from encoder.molgraph_data import molgraph_collate_fn
import argparse
import numpy as np
from EnsembleDataset import EnsembleDataset
from EnsembleModel import EnsembleModel
from torchmetrics.functional import auc, precision_recall
from torchmetrics import Accuracy
from torchmetrics import F1Score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import copy

MODEL_CONSTRUCTOR_DICTS = {
    'AMDE': {
        'constructor': encoder.AMDE_implementations.Graph_encoder,
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 2},
            'message-size': {'type': int, 'default': 25},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 50},
            'att-depth': {'type': int, 'default': 2},
            'att-hidden-dim': {'type': int, 'default': 50},
            'gather-width': {'type': int, 'default': 75},
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 45},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 26},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 90},
            'out-layer-shrinkage': {'type': float, 'default': 0.6}
        }
    }
}


def get_auc_one(pred, target, index):
    pred = copy.copy(pred)
    pred[(pred != index).flatten()] = -1
    pred[(pred == index).flatten()] = 1
    target = copy.copy(target)
    target[(target != index).flatten()] = -1
    target[(target == index).flatten()] = 1
    if len(pred) == 0 or len(set([str(vi) for vi in target])) == 1:
        if np.sum(pred == target) > 0:
            print("batch single category")
            return 0.0
        return 0.0
    return metrics.roc_auc_score(target, pred)


def get_recall_one(pred, target, index):
    pred = copy.copy(pred)
    pred[(pred != index).flatten()] = -1
    pred[(pred == index).flatten()] = 1
    target = copy.copy(target)
    target[(target != index).flatten()] = -1
    target[(target == index).flatten()] = 1
    if len(pred) == 0 or len(set([str(vi) for vi in target])) == 1:
        if np.sum(pred == target) > 0:
            print("batch single category")
            return 0.0
        return 0.0
    return metrics.recall_score(target, pred)


def get_auc(probas_pred, target, classes):
    pred = (probas_pred.detach().cpu().numpy()).astype(int).flatten()
    target = target.detach().cpu().numpy().astype(int).flatten()
    print('\n\nconfusion:\n')
    print(confusion_matrix(target, pred))
    print('\n')
    avg_auc = 0
    avg_recall = 0
    for k, v in classes.items():
        auc = get_auc_one(pred, target, v)
        recall = get_recall_one(pred, target, v)
        avg_auc += auc
        avg_recall += recall
        print(f"auc ({k:3}): {auc:4}")
    avg_auc = avg_auc / len(classes)
    avg_recall = avg_recall / len(classes)
    print(f"auc (avg): {avg_auc:4}", f"recall (avg): {avg_recall:4}")


def main():
    common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    common_args_parser.add_argument('--train-set', type=str, default='./Data/drugbank_train_0.csv',
                                    help='Training dataset path')
    common_args_parser.add_argument('--valid-set', type=str, default='./Data/drugbank_val_0.csv',
                                    help='Validation dataset path')
    common_args_parser.add_argument('--test-set', type=str, default='./Data/test.csv',
                                    help='Testing dataset path')
    common_args_parser.add_argument('--loss', type=str, default='CrossEntropy',
                                    choices=[k for k, v in LOSS_FUNCTIONS.items()])
    common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')
    common_args_parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    common_args_parser.add_argument('--batch-size', type=int, default=64, help='Number of graphs in a mini-batch')
    common_args_parser.add_argument('--learn-rate', type=float, default=1e-4)
    common_args_parser.add_argument('--savemodel', action='store_true', default=False,
                                    help='Saves model with highest validation score')
    common_args_parser.add_argument('--logging', type=str, default='less')
    common_args_parser.add_argument('--model', type=str, default='AMDE')

    main_parser = common_args_parser
    subparsers = main_parser.add_subparsers(help=', '.join([k for k, v in MODEL_CONSTRUCTOR_DICTS.items()]),
                                            dest='model')

    model_parsers = {}
    for model_name, constructor_dict in MODEL_CONSTRUCTOR_DICTS.items():
        subparser = subparsers.add_parser(model_name, parents=[common_args_parser])
        for hp_name, hp_kwargs in constructor_dict['hyperparameters'].items():
            subparser.add_argument('--' + hp_name, **hp_kwargs, help=model_name + ' hyperparameter')
        model_parsers[model_name] = subparser
    args = main_parser.parse_args()
    args_dict = vars(args)
    model_hp_kwargs = {
        name.replace('-', '_'): MODEL_CONSTRUCTOR_DICTS['AMDE']['hyperparameters'][name].items()
        for name, v in MODEL_CONSTRUCTOR_DICTS['AMDE']['hyperparameters'].items()
    }

    train_dataset = EnsembleDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=molgraph_collate_fn)

    validation_dataset = EnsembleDataset(args.valid_set, classes=train_dataset.classes)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,
                                       collate_fn=molgraph_collate_fn)

    num_classes = len(train_dataset.classes)
    ((sample_adj_1, sample_nd_1, sample_ed_1), (sample_adj_2, sample_nd_2, sample_ed_2), sample_target, d1, d2, mask1,
     mask2) = train_dataset[0]['amde']

    # use_seq,use_graph to control two channels, use_ssi,use_3dgt for another two. amde_model to None for not using amde.
    amde_model = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
        node_features_1=len(np.array(sample_nd_1[0])), edge_features_1=len(np.array(sample_ed_1[0, 0])),
        node_features_2=len(np.array(sample_nd_2[0])), edge_features_2=len(np.array(sample_ed_2[0, 0])),
        out_features=86, num_class=num_classes, use_seq=True, use_graph=True,
    )
    ensemble_model = EnsembleModel(amde_model, num_class=num_classes,

                                   n_atom_feats=train_dataset.TOTAL_ATOM_FEATS, use_ssi=True, use_3dgt=True)

    optimizer = optim.Adam(ensemble_model.parameters(), lr=args.learn_rate)

    criterion = torch.nn.functional.nll_loss

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    metric_acc = Accuracy(num_classes=num_classes).to(device)

    f1score = F1Score(num_classes=num_classes).to(device)

    ensemble_model = ensemble_model.to(device)
    print(device)
    for epoch in range(args.epochs):
        print(f"epoch {epoch}\n")
        ensemble_model.train()
        avg_loss = 0
        for i_batch, batch in enumerate(train_dataloader):
            print(f"\rbatch: {i_batch}", sep=' ', end='', flush=True)
            optimizer.zero_grad()
            output = ensemble_model(batch, device)
            targets = torch.as_tensor(batch[-4].y).to(device)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_value_(ensemble_model.parameters(), 5.0)
            optimizer.step()
            avg_loss += loss.detach().item()

        print('\naverage loss:' + str(avg_loss / (i_batch + 1)))
        with torch.no_grad():
            ensemble_model.eval()
            pre = None
            y = None
            for i_batch, batch in enumerate(validation_dataloader):
                print(f"\rbatch validataion: {i_batch}", sep=' ', end='', flush=True)
                output = ensemble_model(batch, device)
                targets = torch.as_tensor(batch[-4].y).to(device)
                output = output.argmax(dim=1, keepdim=True)
                if pre is None:
                    pre = output
                else:
                    pre = torch.cat((pre, output))
                if y is None:
                    y = targets
                else:
                    y = torch.cat((y, targets))

            x = pre.to(torch.int)
            y = y.to(torch.int)
            get_auc(x, y, train_dataset.classes)
            p_r_value = precision_recall(x, y, average='micro')
            p = p_r_value[0].detach().item()
            r = p_r_value[1].detach().item()
            torch_y = torch.unsqueeze(y, 1)
            acc = metric_acc(x, torch_y).detach().item()
            f1s = f1score(x, torch_y).detach().item()
            print(f"val: \t p = {p:4}, \t r = {r:4}, acc = {acc:4} f1 = {f1s:4}\n")
        torch.save(ensemble_model.state_dict(), 'ensemble_model.pt')


if __name__ == '__main__':
    main()
