### AI for Drug Discovery

---

#### Tasks

Broadly there are four tasks [Kang and Cho 2018]:

1. Property prediction (usually QED which measures drug-likeness) 
2. Molecule generation (generate a new structure) [Gómez-Bombarelli et. al. 2018]
3. Targeted molecule generation (to increase some property) [Gómez-Bombarelli et. al. 2018]
4. Synthesis of molecules (pick molecules to combine) [Molecule Chef, Bradshaw et. al. 2019]

Machine learning has historically been used for (1), and the invention of deep latent-variable generative models (i.e. VAEs) has recently led to success at tasks (2) and (3). On the other hand (4) is a new task that makes the most sense from the perspective of the specific domain and it's surprising that nobody has worked no that problem yet.

Throughout the rest of this document we will focus on tasks (2) and (3), which have had the most attention from the computer science community.

#### Technical Approaches

We can break down technical approaches based on the representation of molecules. The following is categorized in order of theoretical satisfaction of the approach. After we discuss a few technical choices in which we may be able to make novel contributions.

1. SMILES-based 
   1. Raw SMILES [Gómez-Bombarelli et. al. 2018, Popova et. al. 2018]
   2. Account for some grammar [GrammarVAE, Kusner et. al. 2017]
2. Graph-based
   1. Sequence representation [CGVAE, Liu et. al. 2018, GraphRNN, You et. al. 2018]
   2. Adjacency matrix representation [GVAE, Kipf and Welling 2016, Graphite, Grover et. al. 2019]
   3. Compositional representation [Junction Tree VAE, Jin et. al. 2018, GCPN You et. al. 2018]

**SMILES vs Graphs** 

The main problem with SMILES-based approaches in comparison to graph-based approaches is that they may not encode valid molecules. However, benchmarks (see next section) have shown that SMILES-based approaches remain competitive with graph-based approaches.

**Permutation Invariance**

One problem that pretty much every approach has had so far has to do with the representation of the molecule graph. For any permutation matrix $P$ (i.e. each row and column has exactly one $1$ entry and all others are $0$):
$$
p_\theta(PAP^\top) = p_\theta(A), \text{for all permutation matrices }P.
$$
Neither SMILES nor graph-based approaches guarantee this invariance. To the best of our knowledge SMILES-based approaches ignore this property altogether and graph-based approaches use Monte Carlo samples of permutations to try to average out the permutation choice (see for example, [GraphRNN You et. al. 2018]).

**Latent variable model**

Pretty much all of the above tasks use VAEs as the generative latent-variable model to generate graphs. We have been exploring the use of normalizing flows instead, which present the advantage of tractably yielding exact likelihoods instead of an evidence lower bound. To summarize, the graphical models look like:
$$
\text{VAE}: z \rightarrow x \quad\quad \text{Flow}: z \leftrightarrow x
$$
Recent work [GNF, Liu et. al. 2019] has constructed a flow for a set of embeddings on a graph assuming a pre-specified graph structure. However, this does not construct a flow for the structure of the graph (i.e. the adjacency matrix), so they dodge around the issue of permutation invariance. Instead their paper proposes a separate encoder to generate and adjacency matrix from a set of node embeddings, and we have had difficulty reproducing their results. We spent most of this week looking into this possibility based on spectral embeddings of nodes that may be able to incorporate information about structure [Verma and Zhang, 2017] but this hasn't worked too well so far. 

#### Benchmarks and Datasets

For the most part advances have relied on ad-hoc experimental validation. Recently there have been two benchmarks for comparing (mostly outdated) that have been released:

1. [[GuacaMol, Brown et. al. 2019]](https://github.com/BenevolentAI/guacamol): 1M datapoints.
2. [[MOSES]](https://github.com/molecularsets/moses): 3M datapoints.

GuacaMol uses a cleaned up version of the ChEMBL 24 dataset, and evaluates for tasks (2) and (3). They found that the SMILES LTSM (without a latent space) is a simple but highly performant baseline model.

MOSES uses the ZINC database, which has been criticized for being a bit biased. It evaluates task (2) only. They found that the SMILES LSTM from [Gómez-Bombarelli et. al. 2018] is highly performant.

---

# Normalizing Flows for Graphs Proposal [OUTDATED PLEASE IGNORE]

### Idea

We follow the notation of [Grover et. al. 2019], repeated below. 

Let $G=(V,E)$ denote a weighted undirected graph which we represent with a set of node features $X \in \mathbb{R}^{n,m}$ and an adjacency matrix $A \in \{0,1\}^{n,n}$, where $n=|V|$. 

We suppose $n$ is fixed, though later on this can easily be modeled as sampled from some parametric distribution. Another modeling possibility is to add discrete labels onto the edges by modeling $A\in \{0,1,\dots,L\}^{n,n}$ but we ignore that for now for the sake of simplicity.

We can factorize a probabilistic model of a graph into,
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

The adjacency matrix is a discrete object. We can instead represent $A$ using de-quantized logits,
$$
\tilde{A} = \sigma^{-1}(A + \epsilon),
$$
where $\epsilon$ is appropriate noise. The flow process therefore looks like the below.
$$
Z_1 \sim N(0, \sigma^2I)\quad\quad \tilde{A} = f_\theta(Z_1)f_\theta(Z_1)^\top
$$


Note that graphs have permutation symmetry; we let $P$ denote a permutation matrix of the same dimensionality as $A$. Ideally we want
$$
p_\theta(A) = p_\theta(P^\top AP), \text{for all permutations }P.
$$
We use a *set transformer* to maintain permutation invariance in the function $f_\theta$.
$$
f_\theta(Z_1) = [...]
$$

---

Note that an alternative approach would be to follow [Kipf and Welling 2016], and use a variational auto-encoder instead of a discrete flow. 
$$
\begin{align*}
	p_\theta(A|Z_1) & = \sigma_\theta(Z_1 Z_1^\top)\\
	q_\phi(Z_1|A) & = g_\phi(Z_1|A).
\end{align*}
$$


However, this does not exhibit permutation invariance and therefore did not work well in practice.

---

#### Node Features

Following [Liu et. al. 2019], we use bipartite (i.e. RealNVP) normalizing flows over a message-passing framework to describe a latent-variable model for $X$. 
$$
Z_2 \sim N(0, \sigma^2I) \quad \quad X|A = f_\theta(Z_2;A).
$$
Using change of variables, the marginal likelihood is then given by
$$
p_\theta(X|A) = p(f_\theta^{-1}(X;A))\left|\det \frac{\partial f_\theta^{-1}(X;A)}{\partial X}\right|.
$$


The message-passing framework yields $p_\theta(X|A)$ that is permutation invariant; i.e.
$$
p_\theta(X|A) = p_\theta(\Gamma X | \Gamma A), \text{for all permutations }\Gamma.
$$
