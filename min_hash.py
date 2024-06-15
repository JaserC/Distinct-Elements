import numpy as np
import mmh3

class MinHash:
    def __init__(self, seed_offset:int=0):
        self.seed_offset = seed_offset
        self.val = float("inf")

    def hash(self, x:int) -> float:
        large_num = 2 ** 31
        h = mmh3.hash(x, self.seed_offset) % large_num + 1
        return h / large_num

    def update(self, x:int):
        self.val = min(self.val, self.hash(x))

    def estimate(self) -> int:
        return round((1/self.val))

class MultMinHash:
    def __init__(self, num_reps:int=1):

        self.num_reps = num_reps
        self.des = [MinHash(seed_offset=i) for i in range(num_reps)]

    def update(self, x:int):
        for minHeap in self.des:
            minHeap.update(x)

    def estimate(self) -> int:

        vals = [minHash.val for minHash in self.des]
        min_estimate = np.mean(vals)
        return round(1/min_estimate) - 1

if __name__ == '__main__':
    stream = np.genfromtxt('data/stream_small.txt', dtype='int')

    print("True Dist Elts: {}".format(312))

    de = MinHash()
    for x in stream:
        de.update(x)
    print("Min Hash Estimate: {}".format(de.estimate()))

    num_reps = 50
    mde = MultMinHash(num_reps=num_reps)
    for x in stream:
        mde.update(x)
    print("Mult Min Hash Estimate with {} copies: {}".format(num_reps, mde.estimate()))
