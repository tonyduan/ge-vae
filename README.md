# Normalizing Flows for Graphs

### Notes

We follow the notation of [Grover et. al. 2019]. 

Let $G=(V,E)$ denote a weighted undirected graph which we represent with a set of node features $X \in \mathbb{R}^{n,m}$ and an adjacency matrix $A \in \mathbb{R}^{n,n}$, where $n=|V|$.

The graph normalizing flow factorizes,
$$
p(G) = p(n)[p(G |n)
$$
