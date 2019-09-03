import json
import pandas as pd
from dfply import *


if __name__ == "__main__":
    
    #datasets = ["community_big", "ego", "grid_big", "protein"]
    datasets = ["ego", "grid_big", "protein"]
    df = []

    for dataset in datasets:
        stats = json.loads(open(f"./ckpts/{dataset}/gf/stats.json", "r").read())
        stats["dataset"] = dataset
        df.append(stats)

    df = pd.DataFrame(df)
    df = df >> select(X.dataset, X.degree, X.cluster, X.orbit)
                      #X.node_bpd_mean, X.node_bpd_stderr, 
                      #X.edge_bpd_mean, X.edge_bpd_stderr)
    print(df)
    print("=" * 79)
    print(df.round(3).to_latex(index = False))

