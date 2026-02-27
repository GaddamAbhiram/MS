import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, SAGEConv, GraphConv, GATConv

class GraphTopoAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 topo_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual=False,
                 concat=True,
                 last_layer=False,
                 prune_ratio=0.713,
                 use_adaptive_smooth=False,
                 use_neighbor_smooth=False,
                 use_fusion=False,
                 return_smooth=False,
                 signed_smooth=False):
        super(GraphTopoAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.prune_ratio = prune_ratio
        self.use_adaptive_smooth = use_adaptive_smooth
        self.use_neighbor_smooth = use_neighbor_smooth
        self.use_fusion = use_fusion
        self.return_smooth = return_smooth
        self.signed_smooth = signed_smooth
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        if self.use_fusion and topo_dim != in_dim:
            self.fusion_t = nn.Linear(topo_dim, in_dim, bias=False)
        else:
            self.fusion_t = None
        if self.use_adaptive_smooth:
            self.smooth_gate = nn.Linear(in_dim, 1)
        else:
            self.smooth_gate = None
        if self.use_fusion:
            self.fusion_gate = nn.Linear(in_dim + topo_dim, in_dim)
        else:
            self.fusion_gate = None

        # weight matrix Wl for leverage property
        if last_layer:
            self.fl = nn.Linear(in_dim+topo_dim, out_dim, bias=False)
        else:
            self.fl = nn.Linear(in_dim, num_heads*out_dim, bias=False)
        # weight matrix Wc for aggregation context
        fc_in_dim = in_dim if self.use_fusion else (in_dim + topo_dim)
        self.fc = nn.Parameter(torch.Tensor(size=(fc_in_dim, num_heads*out_dim)))
        # weight matrix Wq for neighbors' querying
        self.fq = nn.Parameter(torch.Tensor(size=(in_dim, num_heads*out_dim)))
        nn.init.xavier_normal_(self.fl.weight.data)
        nn.init.constant_(self.fc.data, 10e-3)
        nn.init.constant_(self.fq.data, 10e-3)
        self.attn_activation = nn.ELU()
        self.softmax = edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fl = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fl.weight.data)
            else:
                self.res_fl = None
        self.concat = concat
        self.last_layer = last_layer

    def forward(self, inputs, topo):
        # prepare
        h, t = self.feat_drop(inputs), self.feat_drop(topo)  # NxD, N*T
        t_proj = t
        if self.fusion_t is not None:
            t_proj = self.fusion_t(t)
        if self.use_fusion:
            gate = torch.sigmoid(self.fusion_gate(torch.cat((h, t), 1)))
            fused = gate * h + (1 - gate) * t_proj
        else:
            fused = h

        if not self.last_layer:
            ft = self.fl(fused).reshape((fused.shape[0], self.num_heads, -1))  # NxHxD'
            if self.use_fusion:
                ft_c = torch.matmul(fused, self.fc).reshape((fused.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                ft_c = torch.matmul(torch.cat((h, t), 1), self.fc).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            ft_q = torch.matmul(fused, self.fq).reshape((fused.shape[0], self.num_heads, -1))  # NxHxD'
            self.g.ndata.update({'ft' : ft, 'ft_c' : ft_c, 'ft_q' : ft_q})
            self.g.apply_edges(self.edge_attention)
            self.edge_softmax()

            if self.prune_ratio and self.prune_ratio > 0:
                l_s = int(self.prune_ratio * self.g.edata['a_drop'].shape[0])
                if l_s > 0:
                    topk, _ = torch.topk(self.g.edata['a_drop'], l_s, largest=False, dim=0)
                    thd = torch.squeeze(topk[-1])
                    self.g.edata['a_drop'] = self.g.edata['a_drop'].squeeze()
                    self.g.edata['a_drop'] = torch.where(self.g.edata['a_drop']-thd<0, self.g.edata['a_drop'].new([0.0]), self.g.edata['a_drop'])
                    attn_ratio = torch.div((self.g.edata['a_drop'].sum(0).squeeze()+topk.sum(0).squeeze()), self.g.edata['a_drop'].sum(0).squeeze())
                    self.g.edata['a_drop'] = self.g.edata['a_drop'] * attn_ratio
                    self.g.edata['a_drop'] = self.g.edata['a_drop'].unsqueeze(-1)
            
            # multiply source features by attention and aggregate
            self.g.update_all(fn.u_mul_e('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
            ret = self.g.ndata['ft']
            if self.residual:
                if self.res_fl is not None:
                    resval = self.res_fl(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
                else:
                    resval = torch.unsqueeze(h, 1)  # Nx1xD'
                ret = resval + ret
            ret = torch.cat((ret.flatten(1), ft.mean(1).squeeze()), 1) if self.concat else ret.flatten(1)
        else:
            ret = self.fl(torch.cat((h, t), 1))
        if self.use_adaptive_smooth:
            if self.signed_smooth:
                lam = torch.tanh(self.smooth_gate(fused))
            else:
                lam = torch.sigmoid(self.smooth_gate(fused))
            if self.use_neighbor_smooth:
                # Edge-based (neighbor) smoothness with node-wise gate.
                self.g.ndata["h_smooth"] = fused
                self.g.ndata["lam"] = lam
                def _edge_smooth(edges):
                    diff = (edges.src["h_smooth"] - edges.dst["h_smooth"]).pow(2).sum(dim=1)
                    w = 0.5 * (edges.src["lam"].squeeze(-1) + edges.dst["lam"].squeeze(-1))
                    return {"smooth_e": diff * w}
                self.g.apply_edges(_edge_smooth)
                if self.g.num_edges() > 0:
                    # Mean of signed edge losses (can be negative for sharpening when signed_smooth=True)
                    smooth_loss = self.g.edata["smooth_e"].mean()
                else:
                    smooth_loss = fused.new_tensor(0.0)
            else:
                # Topology-alignment smoothness
                # When signed_smooth=True: lam in [-1,1], so term can be negative (rewards divergence / sharpening)
                # Negative smooth_loss will decrease total loss, rewarding feature divergence on heterophilic edges
                smooth_loss = (lam * (fused - t_proj).pow(2)).mean()
        else:
            smooth_loss = ret.new_tensor(0.0)
        if self.return_smooth:
            return ret, smooth_loss
        return ret

    def edge_attention(self, edges):
        c = edges.dst['ft_c']
        q = edges.src['ft_q'] - c
        a = (q * c).sum(-1).unsqueeze(-1)
        return {'a': self.attn_activation(a)}
        
    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        self.g.edata['a_drop'] = self.attn_drop(attention)

class AdaptiveSmoothFusionNet(nn.Module):
    def __init__(
        self,
        g,
        num_layers,
        feats_d,
        feats_t_d,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        residual,
        concat,
        prune_ratio=0.0,
        smooth_decay=1.0,
        feat_residual=False,
        neighbor_smooth=False,
        signed_smooth=False,
        degree_mask_topo=False,
        highfreq_global=False,
    ):
        super().__init__()

        if not isinstance(g, dgl.DGLGraph):
            g = dgl.from_networkx(g)
        self.g = g
        self.uses_topo = True
        self.num_layers = num_layers
        self.activation = activation
        self.concat = concat
        self.smooth_decay = smooth_decay
        self.feat_residual = feat_residual
        self.neighbor_smooth = neighbor_smooth
        self.signed_smooth = signed_smooth
        self.degree_mask_topo = degree_mask_topo
        self.highfreq_global = highfreq_global
        self.feat_res = nn.Linear(feats_d, num_classes, bias=True) if feat_residual else None
        self.feat_res_drop = nn.Dropout(feat_drop) if feat_residual and feat_drop else nn.Identity()

        self.gtn_layers = nn.ModuleList()

        # Input layer
        self.gtn_layers.append(
            GraphTopoAttention(
                g,
                in_dim=feats_d,
                topo_dim=feats_t_d,
                out_dim=num_hidden,
                num_heads=heads[0],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=False,
                concat=concat,
                prune_ratio=prune_ratio,
                use_adaptive_smooth=True,
                use_neighbor_smooth=neighbor_smooth,
                use_fusion=True,
                return_smooth=True,
                signed_smooth=signed_smooth,
            )
        )

        # Hidden layers (keep representation dimension consistent)
        for l in range(1, num_layers + 1):
            in_dim = num_hidden * (heads[l - 1] + (1 if concat else 0))
            self.gtn_layers.append(
                GraphTopoAttention(
                    g,
                    in_dim=in_dim,
                    topo_dim=feats_t_d,
                    out_dim=num_hidden,
                    num_heads=heads[l],
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    residual=residual,
                    concat=concat,
                    prune_ratio=prune_ratio,
                    use_adaptive_smooth=True,
                    use_neighbor_smooth=neighbor_smooth,
                    use_fusion=True,
                    return_smooth=True,
                    signed_smooth=signed_smooth,
                )
            )

        self.post_drop = nn.Dropout(feat_drop) if feat_drop else nn.Identity()

        out_dim = num_hidden * (heads[-1] + (1 if concat else 0))
        self.global_conv = GraphConv(out_dim, out_dim, allow_zero_in_degree=True)
        if highfreq_global:
            self.context_gate = nn.Linear(out_dim * 3, out_dim)
        else:
            self.context_gate = nn.Linear(out_dim * 2, out_dim)
        self.readout = nn.Linear(out_dim, num_classes)

    def forward(self, inputs, topo):
        h = inputs
        t = F.normalize(topo)
        smooth_total = inputs.new_tensor(0.0)

        if self.degree_mask_topo:
            deg = self.g.in_degrees().float().to(h.device)
            deg_mask = (deg > 0).float().unsqueeze(1)
            t = t * deg_mask

        for l, layer in enumerate(self.gtn_layers):
            h, smooth_loss = layer(h, t)
            weight = self.smooth_decay ** l if self.smooth_decay is not None else 1.0
            smooth_total = smooth_total + smooth_loss * weight
            h = self.activation(h)

        global_context = self.global_conv(self.g, h)
        if self.highfreq_global:
            high_freq = h - global_context
            fusion_input = torch.cat([h, global_context, high_freq], dim=1)
            gate = torch.sigmoid(self.context_gate(fusion_input))
            fused = gate * high_freq + (1 - gate) * global_context
        else:
            fusion_input = torch.cat([h, global_context], dim=1)
            gate = torch.sigmoid(self.context_gate(fusion_input))
            fused = gate * h + (1 - gate) * global_context
        fused = self.post_drop(fused)
        logits = self.readout(fused)
        if self.feat_residual:
            logits = logits + self.feat_res(self.feat_res_drop(inputs))
        return logits, smooth_total

class GTN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 feats_d,
                 feats_t_d,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 residual,
                 concat):
        super(GTN, self).__init__()
        self.g = g
        self.uses_topo = True
        self.num_layers = num_layers
        self.gtn_layers = nn.ModuleList()
        self.activation = activation
            
        # input projection (no residual)
        self.gtn_layers.append(GraphTopoAttention(g, feats_d, feats_t_d, num_hidden, heads[0], 
                                                feat_drop, attn_drop, False, concat))
        # hidden layers
        fix_d = concat*(feats_d)
        for l in range(1, num_layers+1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gtn_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d, 
                            num_hidden, heads[l], feat_drop, attn_drop, residual, concat))
        # output projection
        self.gtn_layers.append(GraphTopoAttention(g, num_hidden*(heads[l-1]+1*concat), feats_t_d, 
                num_classes, heads[-1], feat_drop, attn_drop, residual, concat, True))

    def forward(self, inputs, topo):
        h, t = inputs, F.normalize(topo)
        for l in range(self.num_layers+1):
            h = self.gtn_layers[l](h, t)
            h = self.activation(h)
        # output projection
        logits = self.gtn_layers[-1](h, t)
        return logits

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop,
                 aggregator_type="mean"):
        super(GraphSAGE, self).__init__()
        self.g = g
        self.uses_topo = False
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(feat_drop) if feat_drop else nn.Identity()
        self.layers = nn.ModuleList()

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1 for GraphSAGE")

        # input layer
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type))
        # hidden layers
        for _ in range(1, num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(hidden_dim, num_classes, aggregator_type))

    def forward(self, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 activation,
                 feat_drop):
        super(GCN, self).__init__()
        self.g = g
        self.uses_topo = False
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(feat_drop) if feat_drop else nn.Identity()
        self.layers = nn.ModuleList()

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1 for GCN")

        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True))
        # hidden layers
        for _ in range(1, num_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True))
        # output layer
        self.layers.append(GraphConv(hidden_dim, num_classes, allow_zero_in_degree=True))

    def forward(self, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 num_heads,
                 activation,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 residual=False):
        super(GAT, self).__init__()

        # Convert NetworkX graph to DGL if necessary
        if not isinstance(g, dgl.DGLGraph):
            g = dgl.from_networkx(g)

        self.g = g

        # GAT requires self-loops for proper attention computation
        # Add self-loops if they don't exist
        if not self.g.has_edges_between(self.g.nodes(), self.g.nodes()).all():
            self.g = dgl.add_self_loop(self.g)
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(feat_drop)
        self.gat_layers = nn.ModuleList()

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1 for GAT")

        # Input layer
        self.gat_layers.append(GATConv(
            in_dim,
            hidden_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            allow_zero_in_degree=True
        ))

        # Hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GATConv(
                hidden_dim * num_heads,  # Multi-head concatenation
                hidden_dim,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=residual,
                allow_zero_in_degree=True
            ))

        # Output layer (single head, no activation)
        self.gat_layers.append(GATConv(
            hidden_dim * num_heads,
            num_classes,
            num_heads=1,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=False,
            allow_zero_in_degree=True
        ))

    def forward(self, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(self.g, h)
            if i != len(self.gat_layers) - 1:
                # Flatten multi-head outputs
                h = h.flatten(1)
                h = self.activation(h)
                h = self.dropout(h)
            else:
                # Output layer: average multi-head outputs
                h = h.mean(1)
        return h


class SoftGNN(nn.Module):

    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 num_classes,
                 num_layers=2,
                 feat_drop=0.5,
                 activation=F.relu,
                 attn_dim=16):
        super(SoftGNN, self).__init__()

        if not isinstance(g, dgl.DGLGraph):
            g = dgl.from_networkx(g)
        self.g = g
        self.uses_topo = False  # SoftGNN doesn't use topology features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.activation = activation
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop else nn.Identity()
        self.conv_layer_list = []  # Keep track for graph updates

        # Soft Label Predictor components
        # Transform adjacency matrix and features separately
        self.W_A = nn.Linear(in_dim, hidden_dim)
        self.W_X = nn.Linear(in_dim, hidden_dim)
        self.W_T = nn.Linear(2 * hidden_dim, num_classes)

        # Label-Guided Graph Convolution layers
        self.conv_layers = nn.ModuleList()

        # Input layer
        layer = LabelGuidedConvLayer(g, in_dim, hidden_dim, num_classes, attn_dim, feat_drop)
        self.conv_layers.append(layer)
        self.conv_layer_list.append(layer)

        # Hidden layers
        for _ in range(num_layers - 1):
            layer = LabelGuidedConvLayer(g, hidden_dim, hidden_dim, num_classes, attn_dim, feat_drop)
            self.conv_layers.append(layer)
            self.conv_layer_list.append(layer)

        # Final output layer (concatenates all layer outputs)
        self.readout = nn.Linear(hidden_dim * num_layers, num_classes)

    def soft_label_predictor(self, X, A):
   
        # Encode adjacency (topology) information
        H_A = self.activation(self.W_A(A))

        # Encode feature information
        H_X = self.activation(self.W_X(X))

        # Concatenate and transform
        H_combined = torch.cat([H_A, H_X], dim=1)
        P_logits = self.W_T(H_combined)
        P = F.softmax(P_logits, dim=1)

        return P, P_logits

    def forward(self, inputs):

        # Handle both single input and tuple input for compatibility
        if isinstance(inputs, tuple):
            X = inputs[0]
        else:
            X = inputs

        X = self.feat_drop(X)

        # Get adjacency matrix representation for soft label predictor
        # Use a simple aggregation of neighbor features
        with self.g.local_scope():
            self.g.ndata['h'] = X
            self.g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_agg'))
            A_rep = self.g.ndata['h_agg']

        # Step 1: Soft Label Predictor
        soft_labels, soft_label_logits = self.soft_label_predictor(X, A_rep)

        # Step 2: Label-Guided Graph Convolution with soft labels
        h = X
        intermediate_reps = []

        for layer in self.conv_layers:
            h = layer(h, soft_labels)
            h = self.activation(h)
            intermediate_reps.append(h)

        # Step 3: Combine intermediate representations
        h_final = torch.cat(intermediate_reps, dim=1)

        # Step 4: Final classification
        final_logits = self.readout(h_final)

        return final_logits, soft_label_logits


class LabelGuidedConvLayer(nn.Module):

    def __init__(self, g, in_dim, out_dim, num_classes, attn_dim, feat_drop=0.0):
        super(LabelGuidedConvLayer, self).__init__()

        self.g = g  # Store for potential use, but will be set dynamically
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop else nn.Identity()

        # Attention mechanism parameters (operates on aggregated features of in_dim)
        self.W_a = nn.Linear(in_dim, attn_dim)
        self.b = nn.Parameter(torch.zeros(attn_dim))
        self.q = nn.Parameter(torch.randn(attn_dim))
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.q.view(-1, 1))

        # Transformation for ego and neighbor representations
        self.W_t = nn.Linear(in_dim + in_dim, out_dim)

    def forward(self, h, soft_labels):

        # Use the current graph (set externally for minibatching)
        g = self.g
        with g.local_scope():
            # Get degrees for normalization
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1).to(h.device)

            # Aggregate from each class separately
            class_aggregations = []

            for c in range(self.num_classes):
                # Get soft label probability for class c
                p_c = soft_labels[:, c].view(-1, 1)

                # Weight features by soft label probability and apply source normalization
                h_weighted = h * p_c * norm

                # Store weighted features
                g.ndata['h_c'] = h_weighted

                # Simple message passing: each node sums weighted neighbor features
                g.update_all(fn.copy_u('h_c', 'm'), fn.sum('m', 'agg_c'))

                # Apply destination normalization
                agg = g.ndata['agg_c'] * norm
                class_aggregations.append(agg)

            # Stack aggregations from all classes
            class_aggs = torch.stack(class_aggregations, dim=1)  # [N, C, feat_dim]

            # Compute attention weights for each class
            # w_ic = tanh(h_hat_ic * W_a + b) · q
            attn_input = self.W_a(class_aggs) + self.b  # [N, C, attn_dim]
            attn_scores = torch.tanh(attn_input) @ self.q  # [N, C]
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [N, C, 1]

            # Weighted combination of class-wise aggregations
            h_neighbor = (class_aggs * attn_weights).sum(dim=1)  # [N, feat_dim]

            # Ego- and neighbor-representation separation
            h_combined = torch.cat([h, h_neighbor], dim=1)
            h_out = self.W_t(h_combined)

            return h_out

