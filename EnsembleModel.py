import torch

from t3dgt.model import myModel_graph_sch_cnn
from ssi.models import SSI_DDI


class InnterBatch():
    def __init__(self):
        self.pos1 = None
        self.z1 = None
        self.pos1_batch = None
        self.pos2 = None
        self.z2 = None
        self.pos2_batch = None


class EnsembleModel(torch.nn.Module):
    def __init__(self, model_amde, n_atom_feats, num_class=2, n_atom_hid=64, kge_dim=64, rel_total=86,
                 g_out_channels=32, use_3dgt=True, use_ssi=True):
        super(EnsembleModel, self).__init__()
        self.amde = model_amde
        self.m3dgt = None
        self.out_pre = None
        self.ssi = None
        npre = 0
        if self.amde is not None:
            npre = num_class
        if use_3dgt:
            self.m3dgt = myModel_graph_sch_cnn(num_class=num_class, g_out_channels=g_out_channels)
            npre += num_class
        if use_ssi:
            self.ssi = SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32],
                               blocks_params=[2, 2, 2, 2], num_class=num_class)
            npre += num_class
        if npre > 1:
            self.out_pre = torch.nn.Sequential(
                torch.nn.Linear(npre, npre * 4, bias=True),
                torch.nn.PReLU(),
                torch.nn.Linear(npre * 4, num_class, bias=True)
            )

    def forward(self, x, device='cuda'):
        # nn.Linear(32, 1)
        adj_1, nd_1, ed_1, adj_2, nd_2, ed_2, target, d1, d2, mask_1, mask_2, t3dgt_data, ssi_h, ssi_t, ssi_types = x
        pre = None
        if self.amde is not None:
            adj_1 = adj_1.to(device)
            nd_1 = nd_1.to(device)
            ed_1 = ed_1.to(device)
            adj_2 = adj_2.to(device)
            nd_2 = nd_2.to(device)
            ed_2 = ed_2.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)
            mask_1 = mask_1.to(device)
            mask_2 = mask_2.to(device)
            target = target.to(device)
            pre = self.amde(adj_1, nd_1, ed_1, adj_2, nd_2, ed_2, d1, d2, mask_1, mask_2)
        if self.m3dgt is not None:
            t3dgt_data = t3dgt_data.to(device)
            t3dgt_pre = self.m3dgt(t3dgt_data.pos1, t3dgt_data.z1, t3dgt_data.pos1_batch, t3dgt_data.pos2,
                                   t3dgt_data.z2, t3dgt_data.pos2_batch)
            if pre is None:
                pre = t3dgt_pre
            else:
                pre = torch.cat((t3dgt_pre, pre), 1)
        if self.ssi is not None:
            ssi_h = ssi_h.to(device)
            ssi_t = ssi_t.to(device)
            ssi_types = ssi_types.to(device)
            ssi_pre = self.ssi((ssi_h, ssi_t, ssi_types))
            if pre is None:
                pre = ssi_pre
            else:
                pre = torch.cat((ssi_pre, pre), 1)
        if self.out_pre is not None:
            pre = self.out_pre(pre)  # log_max
        return torch.nn.functional.log_softmax(pre, dim=1)
