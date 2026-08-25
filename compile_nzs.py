import numpy as np
import sys

run = sys.argv[1]
nruns = int(sys.argv[2])
path = sys.argv[3]

nzs = []
for i in range(nruns):
    nzs.append(np.load(path+"nz_samples/nz_"+run+"_"+str(i)+".npy", allow_pickle=True))

nzs = np.array(nzs)
nzs = np.vstack(nzs)
print("Shape: ", nzs.shape)

np.save(path+"nz_samples/nz_compiled_"+run+".npy", nzs)
