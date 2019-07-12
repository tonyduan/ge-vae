# Normalizing Flows for Graphs Proposal

#### Idea

We follow the notation of [Grover et. al. 2019], repeated below. 

Let $G=(V,E)$ denote a weighted undirected graph which we represent with a set of node features $X \in \mathbb{R}^{n,m}$ and an adjacency matrix $A \in \{0,1\}^{n,n}$, where $n=|V|$. 

We suppose $n$ is fixed, though this can easily be modeled as sampled from some parametric distribution. We can add discrete labels onto the edges by modeling $A\in \{0,1,\dots,L\}^{n,n}$ but ignore that for now for the sake of simplicity.

We can factorize a probabilistic model of the graphs into,
$$
\begin{align*}
	p(X,A) & = p(A)p(X|A).
\end{align*}
$$

Ideally we want to learn a latent representation $Z_1$ for the graph structure $A$ and a latent representation $Z_2$ for the graph nodes $X$ (such that structure and node features are disentangled). The graphical model looks like
$$
Z_1 \leftrightarrow A \rightarrow X \leftrightarrow Z_2.
$$
We now describe how to model the latent representations $Z_1,Z_2$.

---

#### Adjacency Matrix

Following [Tran et. al. 2019], we want

An alternative approach would be to follow [Kipf and Welling 2016], and use a variational auto-encoder instead of a discrete flow. This would have the advantage of giving us continuous 
$$
\begin{align*}
	p_\theta(A|Z_1) & = \sigma_\theta(Z_1 Z_1^\top)\\
	q_\phi(Z_1|A) & = g_\phi(Z_1|A).
\end{align*}
$$

---

#### Node Features

Following [Liu et. al. 2019], we use bipartite (i.e. RealNVP) normalizing flows over a message-passing framework to describe a latent-variable model for $X$. 
$$
X|A = f_\theta(Z_2;A).
$$
Using change of variables, the marginal likelihood is then given by
$$
p_\theta(X|A) = p(f_\theta^{-1}(X;A))\left|\det \frac{\partial f_\theta^{-1}(X;A)}{\partial X}\right|.
$$
The message-passing framework yields $p_\theta(X|A)$ that is permutation invariant; i.e.
$$
p_\theta(X|A) = p_\theta(\Gamma X | A), \text{for all permutations }\Gamma.
$$
