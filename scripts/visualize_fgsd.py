import numpy as np
from matplotlib import pyplot as plt
from gf.utils import compute_fgsd_embeddings
from datasets.community.gen_community import gen_graphs


if __name__ == "__main__":
    
    A = gen_graphs([10, 20, 30, 40, 50])
    E = [compute_fgsd_embeddings(a) for a in A]

    plt.figure(figsize=(10, 4))
    for i, e in enumerate(E):
        plt.subplot(2, 3, i + 1)
        plt.imshow(e, vmin = -1., vmax = 1.)
        plt.colorbar()
    plt.show()
    plt.savefig("./ckpts/img/fgsd.png")

