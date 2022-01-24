# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import logging

import faiss
import numpy as np


logger = logging.getLogger(__name__)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def write_fvecs(outpath, np_arr):
    dim = np_arr.shape[-1]
    with open(outpath, 'wb') as f:
        for data in np_arr:
            f.write((dim).to_bytes(4, byteorder='little'))
            f.write(data.tobytes())

def write_fvecs_append(outpath, np_arr):
    dim = np_arr.shape[-1]
    with open(outpath, 'ab') as f:
        for data in np_arr:
            f.write((dim).to_bytes(4, byteorder='little'))
            f.write(data.tobytes())


class FaissScorerBase(object):
    def __init__(self, 
            fvec_root
            ) -> None:
        super().__init__()

        self.fvec_root = fvec_root
    
    def load_data(self, proportion_for_training=1.0):

        prob = max(0, min(1.0, proportion_for_training))
        sampling = prob < 1.0

        fpath_list = list(sorted(list(glob.glob(os.path.join(self.fvec_root, "*")))))
        
        fvec_list = []
        for fvec_file in fpath_list:
            fvecs = fvecs_read(fvec_file)
            if sampling:
                num_samples = fvecs.shape[0]
                pick_num = int(num_samples*prob)
                fvecs_indice = np.random.choice(num_samples, pick_num, replace=False)
                fvecs = np.take(fvecs, fvecs_indice, axis=0)
            fvec_list.append(fvecs)
        fvec_list = np.concatenate(tuple(fvec_list), axis=0)
        return fvec_list


class FaissScorer(FaissScorerBase):

    def __init__(self, 
            index_path,
            fvec_root="",
            proportion_for_training=1.0,
            index_str="IVF65536,Flat",
            nprobe=4,
            **kwargs,
            ) -> None:
        super(FaissScorer, self).__init__(fvec_root)
        
        self.index_path=index_path
        self.proportion_for_training = proportion_for_training
        
        self.index = self.load_index(index_str)
        self.index.nprobe = nprobe

    def load_index(self, index_str="IVF65536,Flat"):
        if not os.path.isfile(self.index_path):
            data = self.load_data(self.proportion_for_training)
            d = data.shape[-1]
            index = faiss.index_factory(d, index_str, faiss.METRIC_INNER_PRODUCT)
            logger.info('training index...')
            index.train(data)
            logger.info('loading fvecs...')
            data = self.load_data()
            logger.info('adding index...')
            index.add(data)
            faiss.write_index(index, self.index_path)
        
        return faiss.read_index(self.index_path)
    
    def get_topk(self, query_vec, k=4):
        return self.index.search(query_vec, k)
    

class FaissScorerExhaustive(FaissScorerBase):
    def __init__(self, 
            index_path,
            fvec_root="",
            nprobe=1,
            **kwargs,
            ) -> None:
        super(FaissScorerExhaustive, self).__init__(fvec_root)
        self.index_path=index_path

        self.index = self.load_index()
        self.index.nprobe = nprobe

    def load_index(self):
        if not os.path.isfile(self.index_path):
            logger.info('loading fvecs...')
            data = self.load_data()
            d = data.shape[-1]
            logger.info('vector dim: {}'.format(d))
            index = faiss.IndexFlatIP(d)
            logger.info('adding index...')
            index.add(data)
            
            faiss.write_index(index, self.index_path)
        
        return faiss.read_index(self.index_path)
    
    def get_topk(self, query_vec, k=4):
        return self.index.search(query_vec, k)


class FaissScorerExhaustiveGPU(object):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            fvec_root,
            nprobe=1,
            gpu=0,
            **kwargs,
            ) -> None:
        self.gpu = gpu

        self.fpath_list = list(sorted(list(glob.glob(os.path.join(fvec_root, "*")))))
        self.index = self.load_index(gpu)
        self.index.nprobe = nprobe

    def load_index(self, fvec_root, gpu=0):
        # gpu resources
        res = faiss.StandardGpuResources()

        logger.info('loading fvecs...')
        data = [fvecs_read(path) for path in self.fpath_list]
        d = data[0].shape[-1]
        
        logger.info('vector dim: {}'.format(d))
        index_flat = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(res, gpu, index_flat)
        logger.info('adding index...')
        for ds in data:
            index.add(ds)
        
        return index
    
    def get_topk(self, query_vec, k=4):
        return self.index.search(query_vec, k)


class FaissScorerExhaustiveMultiGPU(object):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            fvec_root,
            nprobe=1,
            gpu_list=None,
            **kwargs,
            ) -> None:
        self.fpath_list = list(sorted(list(glob.glob(os.path.join(fvec_root, "*")))))

        self.gpu_list = gpu_list
        if self.gpu_list is None:
            self.gpu_list = list(range(faiss.get_num_gpus()))

        self.index = self.load_index(fvec_root)
        self.index.nprobe = nprobe


    def load_index(self, fvec_root):

        logger.info('loading fvecs...')
        data = [fvecs_read(path) for path in self.fpath_list]
        data = np.concatenate(tuple(data), axis=0)
        d = data.shape[-1]
        
        logger.info('vector dim: {}'.format(d))
        index_flat = faiss.IndexFlatIP(d)
        gmco = faiss.GpuMultipleClonerOptions()
        gmco.shard = True
        index = faiss.index_cpu_to_gpus_list(index_flat, gmco, self.gpu_list)

        logger.info('adding index...')
        index.add(data)
        
        return index
    
    def get_topk(self, query_vec, k=4):
        return self.index.search(query_vec, k)





