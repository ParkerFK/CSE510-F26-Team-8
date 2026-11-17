from pathlib import Path

import time
import faiss
import numpy as np

DIM_BITS = 768
DIM_BYTES = DIM_BITS // 8


def pack_vector(vec768):
    vec768 = np.asanyarray(vec768, dtype=np.uint8).reshape(-1)
    
    if vec768.size != DIM_BITS:
        raise ValueError(f"Expected {DIM_BITS} bits, but got {vec768.size}")
    
    return np.packbits(vec768).reshape(1, DIM_BYTES)


def load_packed_vectors(vec_path, chunk=100_000):
    src = np.load(vec_path, mmap_mode="r")
    out = np.empty((src.shape[0], DIM_BYTES), dtype=np.uint8)
    
    for start in range(0, src.shape[0], chunk):
        out[start:start + chunk] = np.packbits(np.asarray(src[start:start+chunk]), axis=1)
        
    return out


def load_ids(id_path):
    return np.load(id_path)


def build_flat(packed): 
    index = faiss.IndexBinaryFlat(DIM_BITS)
    index.add(packed)
    return index


def search_flat(index, ids, query, k):
    D, I = index.search(query, k)
    keep = I[0] >= 0
    
    return [str(ids[i]) for i in I[0][keep]], D[0][keep].tolist()


def default_nlist(n):
    return max(16, int(4 * np.sqrt(n)))


def build_ivf(packed, nlist=None, nprobe=8, train_size=100_000, seed=0):
    n = packed.shape[0]
    nlist = nlist or default_nlist(n)
    
    quantizer = faiss.IndexBinaryFlat(DIM_BITS)
    index = faiss.IndexBinaryIVF(quantizer, DIM_BITS, nlist)
    
    rng = np.random.default_rng(seed)
    if n > train_size:
        sample = packed[np.sort(rng.choice(n, train_size, replace=False))]
    else:
        sample = packed
        
    index.train(sample)
    index.add(packed)
    index.nprobe = nprobe
    index._quantizer_ref = quantizer
    
    return index


def search_ivf(index, ids, query, k, nprobe=None):
    params = faiss.IVFSearchParameters()
    params.nprobe = nprobe or index.nprobe
    
    D, I = index.search(query, k, params=params)
    keep = I[0] >= 0
    
    return [str(ids[i]) for i in I[0][keep]], D[0][keep].tolist()


def build_hnsw(packed, M=32, ef_construction=200, ef_search=64):
    index = faiss.IndexBinaryHNSW(DIM_BITS, M)
    index.hnsw.efConstruction = ef_construction
    index.add(packed)
    index.hnsw.efSearch = ef_search
    
    return index


def search_hnsw(index, ids, query, k, ef_search=None):
    params = faiss.SearchParametersHNSW()
    params.efSearch = ef_search or index.hnsw.efSearch
    
    D, I = index.search(query, k, params=params)
    keep = I[0] >= 0
    
    return [str(ids[i]) for i in I[0][keep]], D[0][keep].tolist()


class PuzzleIndexes:
    def __init__(self, ids, indexes, params=None):
        self.ids = ids
        self.indexes = indexes
        self.n = len(ids)
        
        self.params = params or {}
        self.nprobe = self.params.get("nprobe", 8)
        self.ef_search = self.params.get("ef_search", 64)
        
        
        
    @classmethod
    def build(cls, packed, ids, nlist=None, nprobe=8, M=32, ef_construction=200, ef_search=64):
        if packed.shape[0] != len(ids):
            raise ValueError("vectors and ids have different lengths")
        
        nlist = nlist or default_nlist(len(ids))
        
        indexes, times = {}, {}
        for name, make in (
            ("flat", lambda: build_flat(packed)),
            ("ivf", lambda: build_ivf(packed, nlist, nprobe)),
            ("hnsw", lambda: build_hnsw(packed, M, ef_construction, ef_search)),
        ):
            t0 = time.perf_counter()
            indexes[name] = make()
            times[name] = time.perf_counter() - t0

        obj = cls(ids, indexes)
        obj.nprobe, obj.ef_search = nprobe, ef_search
        obj.params = dict(nlist=nlist, M=M, ef_construction=ef_construction,
                          build_seconds=times)
        return obj
    
    
    def _check(self, name, query):
        if name not in self.indexes:
            raise KeyError(f"index '{name}' not loaded; have {list(self.indexes)}")
        
        q = np.ascontiguousarray(query, dtype=np.uint8)
        if q.ndim == 1:
            q = q.reshape(1, -1)
            
        if q.shape[1] != DIM_BYTES:
            raise ValueError(f"query must be packed to {DIM_BYTES} bytes (use pack_vector)")
        
        return q
    
    
    def _params(self, name, nprobe=None, ef_search=None, sel=None):
        if name == 'flat':
            params = faiss.SearchParameters()
            
        elif name == 'ivf':
            params = faiss.IVFSearchParameters()
            params.nprobe = nprobe or self.nprobe
            
        elif name == 'hnsw':
            params = faiss.SearchParametersHNSW()
            params.efSearch = ef_search or self.ef_search
            
        else:
            raise ValueError(f"unknown index type {name }")
        
        if sel is not None:
            params.sel = sel
        
        return params
        
    
    def search_raw(self, name, query, k, nprobe=None, ef_search=None):
        q = self._check(name, query)
        return self.indexes[name].search(q, k, params=self._params(name, nprobe, ef_search))
    
    
    def search(self, name, query, k, nprobe=None, ef_search=None):
        D, I = self.search_raw(name, query, k, nprobe, ef_search)
        keep = I[0] >= 0
        
        return [str(self.ids[i]) for i in I[0][keep]], D[0][keep].tolist()
    
    
    def save(self, index_dir, n=None):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        n = n or self.n
        
        for name, index in self.indexes.items():
            faiss.write_index_binary(index, str(index_dir / f"{name}_{n}.index"))
        np.save(index_dir / f"idmap_{n}.npy", self.ids)
        
    
    @classmethod
    def load(cls, index_dir, n):
        index_dir = Path(index_dir)
        ids = np.load(index_dir / f"idmap_{n}.npy")
        
        indexes = {}
        for name in ("flat", "ivf", "hnsw"):
            path = index_dir / f"{name}_{n}.index"
            if path.exists():
                indexes[name] = faiss.read_index_binary(str(path))
                
        return cls(ids, indexes)
    
    
    def disk_sizes(self, index_dir, n=None):
        n = n or self.n
        sizes = {}
        
        for name in self.indexes:
            path = Path(index_dir) / f"{name}_{n}.index"
            if path.exists():
                sizes[name] = path.stat().st_size
        
        return sizes
    
    
    def search_overfetch(self, name, query, k, keep_fn, start_factor=4, max_fetch=None, nprobe=None, ef_search=None):
        max_fetch = min(max_fetch or self.n, self.n)
        fetch = min(max(k, 1) * start_factor, max_fetch)
        
        while True:
            D, I = self.search_raw(name, query, fetch, nprobe, ef_search)
            out_ids, out_d = [], []
            
            for d , i in zip(D[0], I[0]):
                if i < 0:
                    continue
                
                pid = str(self.ids[i])
                if keep_fn(pid):
                    out_ids.append(pid)
                    out_d.append(int(d))
                    if len(out_ids) == k:
                        return out_ids, out_d, fetch
            
            if fetch >= max_fetch:
                return out_ids, out_d, fetch
            
            fetch = min(fetch * 2, max_fetch)
            
            
    def search_subset(self, name, query, k, allowed_faiss_ids, nprobe=None, ef_search=None):
        allowed = np.ascontiguousarray(allowed_faiss_ids, dtype=np.int64)
        
        if allowed.size == 0:
            return [], [], "empty"
        
        q = self._check(name, query)
        method = name
        index = self.indexes[name]
        
        if name == "ivf":
            if "flat" not in self.indexes:
                raise RuntimeError("ivf subset search needs the flat index loaded as fallback")
            index, method, name = self.indexes["flat"], "flat(fallback for ivf)", "flat"
            
        sel = faiss.IDSelectorBatch(allowed)
        D, I = index.search(q, k, params=self._params(name, nprobe, ef_search, sel))
        keep = I[0] >= 0
        
        return [str(self.ids[i]) for i in I[0][keep]], D[0][keep].tolist(), method 
    
    
def recall_at_k(truth_I, approx_I, truth_D=None, approx_D=None):
    n_q, k = truth_I.shape
    hits = 0
    
    for qi in range(n_q):
        if truth_D is not None and approx_D is not None:
            kth = truth_D[qi, k-1]
            hits += int(np.sum((approx_I[qi] >= 0) & (approx_D[qi] <= kth)))
        else: 
            hits += len(set(truth_I[qi].tolist()) & set(approx_I[qi].tolist()) - {-1})
            
    return hits / (n_q * k)