from torch.utils import data
from encoder.molgraph_data import smile_to_graph, drug2emb_encoder
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from torch_geometric.data import Data as GDATA
from collections import defaultdict


class EnsembleDataset(data.Dataset):
    def __init__(self, train_set, prediction=False, processed_paths='Data/dicts_drugbank.pt', classes=None):

        self.grap_list = []
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
        self.NOTINDICT = 19
        file = pd.read_csv(train_set, sep=',', quoting=1, engine='python')
        self.smiles1 = file['smile1']
        self.smiles2 = file['smile2']
        if prediction:
            self.targets = np.empty((len(self.smiles1), 1))
        else:
            self.targets = file['label']
        self.types = self.targets

        self.pass_smiles = set()
        self.smile_pos_dict = {}
        self.smile_z_dict = {}

        if not os.path.isfile(processed_paths):
            self.process()
            torch.save((self.pass_smiles, self.smile_pos_dict, self.smile_z_dict), processed_paths)
        self.pass_smiles, self.smile_pos_dict, self.smile_z_dict = torch.load(processed_paths)
        self.dict_smiles = {y: x for x, y in enumerate(self.smile_pos_dict.keys())}

        filters = np.zeros(len(self.smiles1))
        for i in range(len(self.smiles1)):
            if self.smile_pos_dict.get(self.smiles1[i], None) is None or self.smile_pos_dict.get(self.smiles2[i],
                                                                                                 None) is None:
                continue
            filters[i] = 1
        print(filters)
        self.smiles1 = self.smiles1.to_numpy()[filters == 1]
        self.smiles2 = self.smiles2.to_numpy()[filters == 1]
        self.targets = self.targets.to_numpy()[filters==1]
        # self.targets = self.targets[filters == 1]
        self.types = self.types.to_numpy()[filters==1]
        # self.types = self.types[filters == 1]
        self.classes = {}
        self.class_idx = defaultdict(lambda: [])
        if classes is None:
            ind = 0
            for v in self.targets:
                if self.classes.get(v, None) is None:
                    # if self.classes.get(v[0], None) is None:
                    self.classes[v] = ind
                    # self.classes[v[0]] = ind
                    ind += 1
        else:
            self.classes = classes

        for i in range(len(self.targets)):
            self.class_idx[self.targets[i]].append(i)
            # self.class_idx[self.targets[i][0]].append(i)
        self.drug_id_mol_graph_tup = None
        self.MOL_EDGE_LIST_FEAT_MTX = None
        self.TOTAL_ATOM_FEATS = None
        self.setup_ssi()

    def one_of_k_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(self, atom,
                      explicit_H=True,
                      use_chirality=False):

        results = self.one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl',
             'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
             ]) + [atom.GetDegree() / 10, atom.GetImplicitValence(),
                   atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  self.one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                             SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + [atom.GetTotalNumHs()]

        if use_chirality:
            try:
                results = results + self.one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        results = np.array(results).astype(np.float32)

        return torch.from_numpy(results)

    def get_mol_edge_list_and_feat_mtx(self, mol_graph):
        features = [(atom.GetIdx(), self.atom_features(atom)) for atom in mol_graph.GetAtoms()]
        features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
        _, features = zip(*features)
        features = torch.stack(features)

        edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
        undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list

        return undirected_edge_list.T, features

    def setup_ssi(self):

        self.drug_id_mol_graph_tup = [Chem.MolFromSmiles(smiles.strip()) for smiles in self.dict_smiles.keys()]
        self.MOL_EDGE_LIST_FEAT_MTX = [self.get_mol_edge_list_and_feat_mtx(mol) for mol in self.drug_id_mol_graph_tup]
        self.TOTAL_ATOM_FEATS = self.MOL_EDGE_LIST_FEAT_MTX[0][1].shape[-1]

    def __getitem__(self, index):

        s1 = self.smiles1[index]
        s2 = self.smiles2[index]
        adj_1, nd_1, ed_1 = smile_to_graph(s1)
        adj_2, nd_2, ed_2 = smile_to_graph(s2)

        d1, mask_1 = drug2emb_encoder(s1)
        d2, mask_2 = drug2emb_encoder(s2)
        targets = self.targets[index]

        index_h = self.dict_smiles[s1]
        index_t = self.dict_smiles[s2]

        targets = self.classes[targets]
        # targets = self.classes[targets[0]]

        return {"amde": ((adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2), targets, d1, d2, mask_1, mask_2),
                "t3dgt": GDATA(pos1=self.smile_pos_dict[s1], z1=self.smile_z_dict[s1],
                               y=targets,
                               pos2=self.smile_pos_dict[s2], z2=self.smile_z_dict[s2]
                               ),
                'ssi': GDATA(type=self.types[index], edge1=self.MOL_EDGE_LIST_FEAT_MTX[index_h][0],
                             f1=self.MOL_EDGE_LIST_FEAT_MTX[index_h][1],
                             edge2=self.MOL_EDGE_LIST_FEAT_MTX[index_t][0], f2=self.MOL_EDGE_LIST_FEAT_MTX[index_t][1])
                }

    def process(self):

        data_len = len(self.smiles1)

        for i in range(data_len):
            print('Converting SMILES to 3Dgraph: {}/{}'.format(i + 1, data_len))

            smile1 = self.smiles1[i]
            smile2 = self.smiles2[i]
            if self.pass_smiles.__contains__(smile1) and self.pass_smiles.__contains__(smile2):
                self.pass_list.append(i)
                continue

            if self.smile_pos_dict.__contains__(smile1):
                ten_pos1 = self.smile_pos_dict[smile1]
                z1 = self.smile_z_dict[smile1]
            else:
                ten_pos1, z1 = self.get_pos_z(smile1)
                if ten_pos1 == None:
                    continue
                else:
                    self.smile_pos_dict[smile1] = ten_pos1
                    self.smile_z_dict[smile1] = z1
            if self.smile_pos_dict.__contains__(smile2):
                ten_pos2 = self.smile_pos_dict[smile2]
                z2 = self.smile_z_dict[smile2]
            else:
                ten_pos2, z2 = self.get_pos_z(smile2)
                if ten_pos2 == None:
                    continue
                else:
                    self.smile_pos_dict[smile2] = ten_pos2
                    self.smile_z_dict[smile2] = z2

    def get_pos_z(self, smile1):
        m1 = Chem.MolFromSmiles(smile1)

        if m1 is None:
            return None, None

        if m1.GetNumAtoms() == 1:
            return None, None
        m1 = Chem.AddHs(m1)

        ignore_flag1 = 0
        ignore1 = False

        while AllChem.EmbedMolecule(m1) == -1:
            print('retry')
            ignore_flag1 = ignore_flag1 + 1
            if ignore_flag1 >= 5:
                ignore1 = True
                break
        if ignore1:
            return None, None
        AllChem.MMFFOptimizeMolecule(m1)
        m1 = Chem.RemoveHs(m1)
        m1_con = m1.GetConformer(id=0)

        pos1 = []
        for j in range(m1.GetNumAtoms()):
            pos1.append(list(m1_con.GetAtomPosition(j)))
        np_pos1 = np.array(pos1)
        ten_pos1 = torch.Tensor(np_pos1)

        z1 = []
        for atom in m1.GetAtoms():
            if self.atomType.__contains__(atom.GetSymbol()):
                z = self.atomType[atom.GetSymbol()]
            else:
                z = self.NOTINDICT
            z1.append(z)

        z1 = np.array(z1)
        z1 = torch.tensor(z1)
        return ten_pos1, z1

    def __len__(self):
        return len(self.smiles1)
