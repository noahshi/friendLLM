from math import inf
import os
import time
import regex as re
import json
from multiprocessing import Pool
import cupy as cp

def merge_dicts(dicts):
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged

class RegexTokenizer:
    merges = {}
    pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""") # gpt4 regex pattern
    special_tokens = {}
    vocab = {}
    MAX_TOKEN_VALUE = 10000
    
    def __init__(self, merges_file=None, vocab_file=None):
        if merges_file and vocab_file:
            self.merges = self.parseMerges(merges_file)
            self.vocab = self.parseVocab(vocab_file)
        else:
            self.merges = self.train("output.txt", self.MAX_TOKEN_VALUE, True)
            self.vocab = self.create_vocab()
        self.special_tokens = {}
        self.reverse_merges = {v: k for k, v in self.merges.items()}
        self.counter = 0
        
    def parseMerges(self, file):
        f = open(file, "r")
        merge_list = json.load(f)
        f.close()
        merges = {tuple(int(x) for x in key[1:-1].split(", ")): int(value) for key, value in merge_list.items()}
        return merges
    
    def parseVocab(self, file):
        f = open(file, "r")
        vocab_list = json.load(f)
        f.close()
        vocab = {int(key): value.encode("utf-8") for key, value in vocab_list.items()}
        return vocab
    
    def train(self, file, vocab_size, verbose=False) -> dict[tuple[int, int], int]:
        f = open(file, "r", encoding="utf-8")
        text = f.read()
        f.close()
        num_merges = vocab_size - 256 - len(self.special_tokens)
        split = re.findall(self.pattern, text)
        ids = [list(x.encode("utf-8")) for x in split]
        merges = {}
        
        for i in range(num_merges):
            counts = self.gpu_pair_counts(ids)
            # start_time = time.time()
            ndpair = cp.unravel_index(cp.argmax(counts), counts.shape)
            pair = (int(ndpair[0]), int(ndpair[1]))
            # end_time = time.time()
            # print(f"Time to find pair: {end_time - start_time} seconds")
            start_time = time.time()
            ids = self.regex_merge(ids, pair, 256+i)
            merges[pair] = 256+i
            end_time = time.time()
            print(f"Time to merge: {end_time - start_time} seconds")
            if verbose:
                print(f"Merging {pair} into token {256 + i}")
        f = open("merges.json", "w")
        merge_list = {str(key): value for key, value in merges.items()}
        json.dump(merge_list, f)
        f.close()
        return merges
    
    def encode(self, text):
        if len(text) < 1000:
            return self.small_encoder(text)
        else:
            return self.large_encoder(text)
    
    def small_encoder(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            counts = self.get_pair_counts(tokens)
            pair = min(counts, key=lambda x: self.merges.get(x, inf))
            if pair not in self.merges:
                break # done merging
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
    
    def large_encoder(self, text):
        tokens = list(text.encode("utf-8"))
        tokens = cp.array(tokens)
        for i in range(257, self.MAX_TOKEN_VALUE):
            # print(i)
            if len(tokens) <= 1:
                break
            counts = self.simple_pair_counts(tokens)
            
            #start_time = time.time()
            pair = self.reverse_merges.get(i)
            if counts[pair[0], pair[1]] == 0:
                continue
            #end_time = time.time()
            #print(f"Time to find {pair}: {end_time - start_time} seconds")
            
            #start_time = time.time()
            tokens = self.cupy_merge(tokens, pair, i)
            #end_time = time.time()
            #print(f"Time to merge {pair}: {end_time - start_time} seconds")
        
        tokens = tokens.tolist()
        
        f = open("bytes{self.counter}.json", "w")
        json.dump(tokens, f)
        f.close()
        self.counter += 1
        return tokens
            
    def decode(self, ids):
        tokens = b"".join([self.vocab[i] for i in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def get_pair_counts(self, ids : list[int]):
        pairs = {}
        for pair in zip(ids, ids[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1
        
        # top_pair = max(pairs, key=pairs.get)
        # print(type(top_pair))
        return pairs
    
    def gpu_pair_counts(self, ids):
        pairs = cp.zeros((self.MAX_TOKEN_VALUE, self.MAX_TOKEN_VALUE), dtype=cp.int32)

        # Flatten the list of lists into a single list
        flat_ids = [item for sublist in ids for item in sublist]
        sublist_sizes = [len(sublist) for sublist in ids]
        flat_ids_gpu = cp.array(flat_ids, dtype=cp.int32)
        boundaries_gpu = cp.cumsum(cp.array(sublist_sizes, dtype=cp.int32))  # Compute boundaries as cumulative sum of sublist sizes
        
        # Define a CUDA kernel for counting pairs
        add_kernel_code = '''
        extern "C" __global__
        void count_pairs(const int *flat_ids, int *pairs, int num_elements, int max_value) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_elements - 1) {
                int i = flat_ids[idx];
                int j = flat_ids[idx + 1];
                atomicAdd(&pairs[i * max_value + j], 1);
            }
        }
        '''
        # Define a CUDA kernel for subtracting boundary pairs
        sub_kernel_code = '''
        extern "C" __global__
        void subtract_boundary_pairs(const int *flat_ids, int *pairs, const int* boundaries, int num_boundaries, int max_value) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_boundaries) {
                int boundary = boundaries[idx] - 1;
                if (boundary >= 0) {
                    int i = flat_ids[boundary];
                    int j = flat_ids[boundary + 1];
                    atomicSub(&pairs[i * max_value + j], 1);
                }
            }
        }
        '''
        
        # Compile and get the kernel
        count_pairs_kernel = cp.RawKernel(add_kernel_code, 'count_pairs')
        subtract_boundary_pairs_kernel = cp.RawKernel(sub_kernel_code, 'subtract_boundary_pairs')
        
        time_start = time.time()
        
        # Prepare grid and block dimensions
        num_elements = len(flat_ids_gpu)
        block_size = 256
        grid_size = (num_elements + block_size - 1) // block_size
        
        # Launch the kernel
        count_pairs_kernel(
            (grid_size,), (block_size,),
            (flat_ids_gpu, pairs, num_elements, self.MAX_TOKEN_VALUE)
        )
        
        cp.cuda.Device().synchronize()  # Wait for the GPU to finish
        
        # Prepare grid and block dimensions
        num_boundaries = len(boundaries_gpu)
        block_size = 256
        grid_size = (num_boundaries + block_size - 1) // block_size
        
        # Launch the kernel
        subtract_boundary_pairs_kernel(
            (grid_size,), (block_size,),
            (flat_ids_gpu, pairs, boundaries_gpu, num_boundaries, self.MAX_TOKEN_VALUE)
        )
        
        cp.cuda.Device().synchronize()  # Wait for the GPU to finish
        time_end = time.time()
        print(f"Time to count pairs: {time_end - time_start} seconds")
        
        return pairs
    
    def simple_pair_counts(self, ids):
        pairs = cp.zeros((self.MAX_TOKEN_VALUE, self.MAX_TOKEN_VALUE), dtype=cp.int32)

        # Define a CUDA kernel for counting pairs
        add_kernel_code = '''
        extern "C" __global__
        void count_pairs(const int *flat_ids, int *pairs, int num_elements, int max_value) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_elements - 1) {
                int i = flat_ids[idx];
                int j = flat_ids[idx + 1];
                atomicAdd(&pairs[i * max_value + j], 1);
            }
        }
        '''
        # Compile and get the kernel
        count_pairs_kernel = cp.RawKernel(add_kernel_code, 'count_pairs')
        # time_start = time.time()
        
        # Prepare grid and block dimensions
        num_elements = len(ids)
        block_size = 256
        grid_size = (num_elements + block_size - 1) // block_size
        
        # Launch the kernel
        count_pairs_kernel(
            (grid_size,), (block_size,),
            (ids, pairs, num_elements, self.MAX_TOKEN_VALUE)
        )
        
        cp.cuda.Device().synchronize()  # Wait for the GPU to finish
        # time_end = time.time()
        # print(f"Time to count pairs: {time_end - time_start} seconds")
        
        return pairs
    
    def merge(self, bytes, pair, index):
        new_bytes = []
        i = 0
        while i < len(bytes):
            if i < len(bytes) - 1 and bytes[i] == pair[0] and bytes[i+1] == pair[1]:
                new_bytes.append(index)
                i += 2
            else:
                new_bytes.append(bytes[i])
                i += 1
        return new_bytes
    
    def cupy_merge(self, bytes, pair, index):
        if type(bytes) == list:
            bytes_cp = cp.array(bytes)
        else:
            bytes_cp = bytes

        # Create a mask where the pair is found
        mask = (bytes_cp[:-1] == pair[0]) & (bytes_cp[1:] == pair[1])
        mask = mask.astype(cp.int32)
        
        # Define CUDA kernel for gpu merge
        kernel_code = '''
        extern "C" __global__
        void gpu_merge_kernel(int *input, int *mask, int index, int length) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < length - 1 && mask[idx]) {
                input[idx] = index;
                input[idx + 1] = -1;
            }
        }
        '''
        
        # Compile and get the kernel
        try:
            gpu_merge_kernel = cp.RawKernel(kernel_code, 'gpu_merge_kernel')
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error compiling CUDA kernel: {e}")
            return
        
        # Launch kernel
        block_size = 256
        grid_size = (len(bytes_cp) + block_size - 1) // block_size

        try:
            gpu_merge_kernel(
                (grid_size,), (block_size,),
                (bytes_cp, mask, index, len(bytes_cp))
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error launching CUDA kernel: {e}")
            return
        
        cp.cuda.Device().synchronize() # Wait for the GPU to finish
        
        return bytes_cp[bytes_cp != -1]
    
    def regex_merge(self, bytes_list, pair, index):
        new_bytes = []
        for x in bytes_list:
            new_x = []
            i = 0
            while i < len(x):
                if i < len(x) - 1 and x[i] == pair[0] and x[i+1] == pair[1]:
                    new_x.append(index)
                    i += 2
                else:
                    new_x.append(x[i])
                    i += 1
            new_bytes.append(new_x)
        return new_bytes
    
    # FIX: This function is not working properly
    def gpu_merge(self, bytes_list, pair, index, batch_size=1000):
        # Convert pair and index to GPU arrays
        pair0, pair1 = pair
        
        # Flatten bytes_list into a single list
        flat_bytes = [item for sublist in bytes_list for item in sublist]
        length = len(flat_bytes)
        
        # Define CUDA kernel for gpu merge
        kernel_code = '''
        extern "C" __global__
        void gpu_merge_kernel(const int *input, int *output, 
                                int pair0, int pair1, int index, int length) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < length) {
                // Check if we are processing an element of a pair
                if (idx < length - 1 && input[idx] == pair0 && input[idx + 1] == pair1) {
                    output[idx] = index;
                    // Skip the next element
                    if (idx + 2 < length) {
                        output[idx + 1] = index;
                        idx += 2;
                    } else {
                        idx += 1;
                    }
                } else {
                    output[idx] = input[idx];
                }
            }
        }
        '''
        
        # Compile and get the kernel
        try:
            gpu_merge_kernel = cp.RawKernel(kernel_code, 'gpu_merge_kernel')
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error compiling CUDA kernel: {e}")
            return

        # Convert the flattened list to a CuPy array
        bytes_gpu = cp.array(flat_bytes, dtype=cp.int32)
        new_bytes_gpu = cp.zeros_like(bytes_gpu, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (length + block_size - 1) // block_size

        try:
            gpu_merge_kernel(
                (grid_size,), (block_size,),
                (bytes_gpu, new_bytes_gpu, pair0, pair1, index, length)
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error launching CUDA kernel: {e}")
            return

        # Transfer results from GPU to CPU
        try:
            new_bytes_cpu = new_bytes_gpu.get()
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error transferring data from GPU to CPU: {e}")
            return

        # Reshape the result back into lists of lists if necessary
        result = []
        start = 0
        for sublist in bytes_list:
            end = start + len(sublist)
            result.append(new_bytes_cpu[start:end].tolist())
            start = end

        return result
    
    def create_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p, q), idx in self.merges.items():
            vocab[idx] = vocab[p] + vocab[q]
        f = open("vocab.json", "w")
        vocab_str = {key: value.decode("utf-8", errors="replace") for key, value in vocab.items()}
        json.dump(vocab_str, f)
        f.close()
        return vocab
    
    def chunk_merge(self, bytes, pair, index):
        new_bytes = []
        for x in bytes:
            new_x = []
            i = 0
            while i < len(x):
                if i < len(x) - 1 and x[i] == pair[0] and x[i+1] == pair[1]:
                    new_x.append(index)
                    i += 2
                else:
                    new_x.append(x[i])
                    i += 1
            new_bytes.append(new_x)
        return new_bytes
    
    def parallel_merge(self, ids, pair, index, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count()
            
        chunk_size = len(ids) // num_workers
        chunks = [ids[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
        
        if len(ids) % num_workers != 0:
            chunks.append(ids[num_workers * chunk_size:])
            
        with Pool(num_workers) as pool:
            results = pool.starmap(self.chunk_merge, [(chunk, pair, index) for chunk in chunks])
        
        new_bytes = [l for sublist in results for l in sublist]
        return new_bytes
    
    def chunk_pair_counts(self, ids):
        pairs = {}
        for i in ids:
            for pair in zip(i, i[1:]):
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def parallel_pair_counts(self, ids, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count()
        
        # create chunks
        chunk_size = len(ids) // num_workers
        chunks = [ids[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
        
        # append extra strings to last chunk
        if len(ids) % num_workers != 0:
            chunks.append(ids[num_workers * chunk_size:])
        
        # start process pool
        with Pool(num_workers) as pool:
            results = pool.map(self.chunk_pair_counts, chunks)
            
        # merge all counts
        pairs = merge_dicts(results)
        return pairs