import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, extract_tar
from urllib.request import urlretrieve
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm


torch_geometric.seed_everything(2022)


def dimacs2list(file, offset=0):
    '''read dimacs 'edge' format into a dict {'n': ..., 'm': ..., 'edges': ...}.
    edges is a list of lists of length 2 and node indexing starts at offset'''
    edges = list()
    for line in file:
        if line[0] == 'e':
            tokens = line.split()
            edges.append([int(tokens[1])-1+offset, int(tokens[2])-1+offset])
        elif line[0] == 'c':
            # skip comments
            continue
        elif line[0] == 'p':
            tokens = line.split()
            if tokens[1].lower() != 'edge':
                raise IOError(f'unknown problem instance: {line} in {file}')
            n = int(tokens[2])
            m = int(tokens[3])
        else:
            raise IOError(f'unknown line format: {line} in {file}')
    return {'n': n, 'm': m, 'edges': edges}


def load_dimacs(file):
    '''return a pytorch_geometric object representing the graph in file'''
    with open(file, 'r') as f:
        g = dimacs2list(f, 0)
        G = nx.Graph(g['edges'])
        return from_networkx(G)


class AachenDataset(InMemoryDataset):
    '''The dataset described in 
    https://www.lics.rwth-aachen.de/cms/LICS/Forschung/Publikationen/~rtok/Benchmark-Graphs/
    
    and the publication

    Pascal Schweizer, Daniel Neuen (2017):
    Benchmark Graphs for Practical Graph Isomorphism
    https://arxiv.org/abs/1705.03686

    '''

    datasets = {
        'cfi-rigid-z2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgajq',
        'cfi-rigid-r2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgalg',
        'cfi-rigid-s2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgaln',
        'cfi-rigid-t2' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgalz',
        'cfi-rigid-z3' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgama',
        'cfi-rigid-d3' : 'https://www.lics.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcgamx',
    }

    def __init__(
        self,
        name="AACHEN",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)
    
    @property
    def raw_dir(self):
        name = "raw"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return [f'{d}.tar.gz' for d in self.datasets]

    @property
    def processed_file_names(self):
        return ['aachen.pt']

    def download(self) -> None:
        for subset in self.datasets:
            print(f'Downloading {subset} from {self.datasets[subset]}')
            urlretrieve(f'{self.datasets[subset]}', os.path.join(self.raw_dir, f'{subset}.tar.gz'))


    def process(self):

        part_dict = dict()
        data_list = list()
        offset = 0

        for f in self.datasets:
            extract_tar(os.path.join(self.raw_dir, f'{f}.tar.gz'), os.path.join(self.raw_dir, 'unzipped'))
            print(f'converting {f}')
            counter = 0
            for data in sorted(os.listdir(os.path.join(self.raw_dir, 'unzipped', f))):
                data_list.append(load_dimacs(os.path.join(self.raw_dir, 'unzipped', f, data)))
                os.remove(os.path.join(self.raw_dir, 'unzipped', f, data))
                counter += 1
            part_dict[f] = (offset, offset + counter)
            offset += counter
            os.rmdir(os.path.join(self.raw_dir, 'unzipped', f))
            
        print(part_dict)
        os.rmdir(os.path.join(self.raw_dir, 'unzipped'))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = AachenDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
