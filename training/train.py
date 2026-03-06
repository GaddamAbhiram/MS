import warnings
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import random
import numpy as np
import time
import dgl
from dgl import DGLGraph
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.utils import load_data
from models import GTN, GraphSAGE, GCN, AdaptiveSmoothFusionNet, GAT, SoftGNN

warnings.filterwarnings("ignore", message="Recommend creating graphs by `dgl.graph")
warnings.filterwarnings("ignore", message="adjacency_matrix will return a scipy.sparse array")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Default seed (can be overridden via --seed argument)
set_seed(123)

def set_model_graph(model, g):
    model.g = g
    if hasattr(model, "gtn_layers"):
        for layer in model.gtn_layers:
            layer.g = g
    if hasattr(model, "conv_layer_list"):  # For SoftGNN
        for layer in model.conv_layer_list:
            layer.g = g

def model_forward(model, feats, feats_t, adj=None, model_name=None, train_mask=None, labels=None):
    if getattr(model, "uses_topo", False):
        out = model(feats.float(), feats_t.float())
    else:
        out = model(feats.float())
    if isinstance(out, tuple):
        return out
    return out, None

def evaluate(g, feats, feats_t, labels, mask, model, loss_fcn, device, model_name=None, adj=None, train_mask=None, gt_labels=None):
    with torch.no_grad():
        model.eval()
        set_model_graph(model, g)
        output, _ = model_forward(model, feats, feats_t, adj=adj, model_name=model_name, train_mask=train_mask, labels=gt_labels)
        loss = loss_fcn(output[mask], labels[mask])
        # Labels are now class indices (not one-hot), so no argmax needed
        predict = np.argmax(output[mask].data.cpu().numpy(), axis=1)
        true = labels[mask].data.cpu().numpy()  # Already class indices
        score = accuracy_score(true, predict)
        return score, loss.item()

def train_main(args):
    # cpu or gpu
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    # create the dataset
    g, train_subgraphs, train_subfeats, train_subfeats_t, train_sublabels, train_submasks, val_subgraphs, val_subfeats, \
    val_subfeats_t, val_sublabels, val_submasks, test_subgraphs, test_subfeats, test_subfeats_t, test_sublabels, \
    test_submasks = load_data(args.prefix, args.num_layers, args.batch_size, args.concat, args.sample_number)

    # compute class weights for balanced loss
    class_weights = None
    if args.class_weighted:
        all_train_labels = []
        for i in range(len(train_subgraphs)):
            all_train_labels.append(train_sublabels[i][train_submasks[i]])
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        class_counts = all_train_labels.sum(axis=0)
        class_counts = np.maximum(class_counts, 1.0)
        total = class_counts.sum()
        class_weights = total / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Class weights: {class_weights}")

    # define the model and optimizer
    if args.model == "gtn":
        heads = ([args.num_heads] * (args.num_layers + 1))
        model = GTN(g,
                    args.num_layers,
                    train_subfeats[0].shape[1],
                    args.sample_number*2,
                    args.num_hidden,
                    train_sublabels[0].shape[1],
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.residual,
                    args.concat)
    elif args.model == "graphsage":
        model = GraphSAGE(g,
                          train_subfeats[0].shape[1],
                          args.num_hidden,
                          train_sublabels[0].shape[1],
                          args.num_layers,
                          F.elu,
                          args.in_drop)
    elif args.model == "gcn":
        model = GCN(g,
                    train_subfeats[0].shape[1],
                    args.num_hidden,
                    train_sublabels[0].shape[1],
                    args.num_layers,
                    F.elu,
                    args.in_drop)
    elif args.model == "asfn":
        heads = ([args.num_heads] * (args.num_layers + 1))
        model = AdaptiveSmoothFusionNet(
            g,
            args.num_layers,
            train_subfeats[0].shape[1],
            args.sample_number * 2,
            args.num_hidden,
            train_sublabels[0].shape[1],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.residual,
            args.concat,
            prune_ratio=args.prune_ratio,
            smooth_decay=args.smooth_decay,
            feat_residual=args.feat_residual,
            neighbor_smooth=args.neighbor_smooth,
            signed_smooth=args.signed_smooth,
            degree_mask_topo=args.degree_mask_topo,
            highfreq_global=args.highfreq_global,
            no_fusion_gate=args.no_fusion_gate,
            no_topo=args.no_topo,
            no_global_fusion=args.no_global_fusion,
        )
    elif args.model == "gat":
        model = GAT(g,
                    train_subfeats[0].shape[1],
                    args.num_hidden,
                    train_sublabels[0].shape[1],
                    args.num_layers,
                    args.num_heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.residual)
    elif args.model == "softgnn":
        softgnn_drop = args.in_drop if args.softgnn_feat_drop < 0 else args.softgnn_feat_drop
        model = SoftGNN(g,
                        train_subfeats[0].shape[1],
                        args.num_hidden,
                        train_sublabels[0].shape[1],
                        args.num_layers,
                        softgnn_drop,
                        F.elu,
                        attn_dim=args.softgnn_attn_dim)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)
    attn_params_name = ['fc', 'fq']
    attn_params = []
    for p in attn_params_name:
        attn_params = attn_params + list(filter(lambda kv: p in kv[0], model.named_parameters()))
    base_params = [param[1] for param in model.named_parameters() if param not in attn_params]
    attn_params = [param[1] for param in attn_params]
    if attn_params:
        optimizer = torch.optim.Adam([{'params': base_params},
                                    {'params': attn_params, 'lr': args.lr/10}],
                                    lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(base_params, lr=args.lr, weight_decay=args.weight_decay)

    # Use CrossEntropyLoss for single-label multi-class classification
    if class_weights is not None:
        loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()

    # start training
    best_score, best_loss, cur_step = 0, 1000, 0
    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        # shuffle
        idx = [i for i in range(len(train_subgraphs))]
        random.shuffle(idx)
        for i in range(len(train_subgraphs)):
            feats = torch.FloatTensor(train_subfeats[idx[i]]).to(device)
            feats_t = torch.FloatTensor(train_subfeats_t[idx[i]]).to(device)
            labels_onehot = torch.FloatTensor(train_sublabels[idx[i]]).to(device)
            # Convert one-hot labels to class indices for CrossEntropyLoss
            labels = torch.argmax(labels_onehot, dim=1)
            set_model_graph(model, train_subgraphs[idx[i]])
            output, auxiliary_output = model_forward(model, feats, feats_t, model_name=args.model)

            # For SoftGNN, auxiliary_output is soft_label_logits
            # For ASFN, auxiliary_output is smooth_loss
            if args.model == "softgnn" and auxiliary_output is not None:
                # SoftGNN dual loss: L = (1-λ) * L_P + λ * L_G
                soft_label_loss = loss_fcn(auxiliary_output[train_submasks[idx[i]]], labels[train_submasks[idx[i]]])
                final_loss = loss_fcn(output[train_submasks[idx[i]]], labels[train_submasks[idx[i]]])
                loss = (1 - args.smooth_lambda) * soft_label_loss + args.smooth_lambda * final_loss
            else:
                loss = loss_fcn(output[train_submasks[idx[i]]], labels[train_submasks[idx[i]]])
                if auxiliary_output is not None:  # ASFN smooth loss
                    loss = loss + args.smooth_lambda * auxiliary_output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        # validation
        score_list = []
        val_loss_list = []
        for i in range(len(val_subgraphs)):
            feats = torch.FloatTensor(val_subfeats[i]).to(device)
            feats_t = torch.FloatTensor(val_subfeats_t[i]).to(device)
            labels_onehot = torch.FloatTensor(val_sublabels[i]).to(device)
            # Convert one-hot labels to class indices for CrossEntropyLoss
            labels = torch.argmax(labels_onehot, dim=1)

            score, val_loss = evaluate(val_subgraphs[i],
                                        feats,
                                        feats_t,
                                        labels,
                                        val_submasks[i],
                                        model,
                                        loss_fcn,
                                        device,
                                        model_name=args.model)
            score_list.append(score)
            val_loss_list.append(val_loss)
        
        # early stop
        if sum(score_list)/len(score_list) > best_score: #or sum(val_loss_list)/len(val_loss_list) < best_loss:
            print("Epoch {:03d} |Train Loss: {:.4f} | Val Loss: {:.4f} | Acc: {:.4f} | save".format(epoch + 1, 
                                                                        sum(loss_list)/len(loss_list), 
                                                                        sum(val_loss_list)/len(val_loss_list),
                                                                        sum(score_list)/len(score_list)))
            torch.save(model.state_dict(), args.prefix.split('/')[-1]+'_best.pkl')
            best_score = sum(score_list)/len(score_list)
            best_loss = sum(val_loss_list)/len(val_loss_list)
            cur_step = 0
            optimizer.param_groups[0]['lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[-1]['lr'] = args.lr/10
        else:
            print("Epoch {:05d} |Train Loss: {:.4f} | Val Loss: {:.4f} | Acc: {:.4f} ".format(epoch + 1, 
                                                                        sum(loss_list)/len(loss_list), 
                                                                        sum(val_loss_list)/len(val_loss_list),
                                                                        sum(score_list)/len(score_list)))
            cur_step += 1
            if cur_step == int(args.patience/2):
                optimizer.param_groups[0]['lr'] = args.lr
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[-1]['lr'] = args.lr
            if cur_step > args.patience:
                break
    # test
    model.load_state_dict(torch.load(args.prefix.split('/')[-1]+'_best.pkl'))
    test_score_list = []
    test_loss_list = []
    for i in range(len(test_subgraphs)):
        feats = torch.FloatTensor(test_subfeats[i]).to(device)
        feats_t = torch.FloatTensor(test_subfeats_t[i]).to(device)
        labels_onehot = torch.FloatTensor(test_sublabels[i]).to(device)
        # Convert one-hot labels to class indices for CrossEntropyLoss
        labels = torch.argmax(labels_onehot, dim=1)

        test_score, test_loss = evaluate(test_subgraphs[i],
                                feats,
                                feats_t,
                                labels,
                                test_submasks[i],
                                model,
                                loss_fcn,
                                device,
                                model_name=args.model)
        test_score_list.append(test_score)
        test_loss_list.append(test_loss)
    print("The test Loss: {:.4f}, Acc: {:.4f}".format(sum(test_loss_list)/len(test_loss_list),
                                                                sum(test_score_list)/len(test_score_list)))

def main():
    parser = argparse.ArgumentParser(description='GTN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--model", type=str, default="asfn",
                        choices=["gtn", "graphsage", "gcn", "asfn", "gat", "softgnn"],
                        help="which model to use")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--concat", action="store_true", default=True,
                        help="concat neighbors with self")
    parser.add_argument("--in-drop", type=float, default=0.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help="weight decay")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=100,
                        help="used for early stop")
    parser.add_argument("--sample-number", type=int, default=32,
                        help="characteristic function sample number, delete feats_t.npy before change")
    parser.add_argument("--smooth-lambda", type=float, default=0.05,
                        help="weight for adaptive smoothness loss")
    parser.add_argument("--prune-ratio", type=float, default=0.0,
                        help="ratio for attention pruning")
    parser.add_argument("--smooth-decay", type=float, default=1.0,
                        help="layer-wise decay for smoothness loss")
    parser.add_argument("--neighbor-smooth", action="store_true", default=False,
                        help="use neighbor-based smoothness loss (ASFN only)")
    parser.add_argument("--feat-residual", action="store_true", default=False,
                        help="use input feature residual in ASFN")
    parser.add_argument("--signed-smooth", action="store_true", default=False,
                        help="use tanh (signed) lambda for smoothness, enabling sharpening on heterophilic edges")
    parser.add_argument("--degree-mask-topo", action="store_true", default=False,
                        help="zero out topology features for isolated (degree-0) nodes")
    parser.add_argument("--highfreq-global", action="store_true", default=False,
                        help="add high-frequency branch to global context fusion")
    parser.add_argument("--no-fusion-gate", action="store_true", default=False,
                        help="disable adaptive fusion gate, fall back to concatenation (ablation)")
    parser.add_argument("--no-topo", action="store_true", default=False,
                        help="zero out topology features (ablation)")
    parser.add_argument("--no-global-fusion", action="store_true", default=False,
                        help="disable global-local context fusion (ablation)")
    parser.add_argument("--class-weighted", action="store_true", default=False,
                        help="use inverse-frequency class weights in BCE loss")
    parser.add_argument("--softgnn-attn-dim", type=int, default=16,
                        help="attention dimension for SoftGNN")
    parser.add_argument("--softgnn-feat-drop", type=float, default=-1.0,
                        help="override SoftGNN feature dropout (default: use --in-drop)")
    parser.add_argument('--prefix', type=str, default='./data/amazon/amazon',
                        help="which dataset to use")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for reproducibility")
    args = parser.parse_args()
    # Re-set seed from command-line argument
    set_seed(args.seed)
    if args.model == "softgnn":
        if args.num_hidden == 32:
            args.num_hidden = 16
        if args.num_layers == 2:
            args.num_layers = 1
        if args.in_drop == 0.3:
            args.in_drop = 0.6
        if args.softgnn_feat_drop < 0:
            args.softgnn_feat_drop = 0.6
        if args.weight_decay == 0.0:
            args.weight_decay = 5e-4
        if args.softgnn_attn_dim == 16:
            args.softgnn_attn_dim = 8
        if args.smooth_lambda == 0.05:
            args.smooth_lambda = 0.2
    print(args)
    train_main(args)


if __name__ == '__main__':
    main()
