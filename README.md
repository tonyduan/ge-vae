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

One problem that pretty much every approach has had so far has to do with the representation of the molecule graph. For any permutation matrix <img alt="$P$" src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg" align="middle" width="12.83677559999999pt" height="22.465723500000017pt"/> (i.e. each row and column has exactly one <img alt="$1$" src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg" align="middle" width="8.219209349999991pt" height="21.18721440000001pt"/> entry and all others are <img alt="$0$" src="svgs/29632a9bf827ce0200454dd32fc3be82.svg" align="middle" width="8.219209349999991pt" height="21.18721440000001pt"/>):
<p align="center"><img alt="$$&#10;p_\theta(PAP^\top) = p_\theta(A), \text{for all permutation matrices }P.&#10;$$" src="svgs/a529e169c83fca150360b89cc1654833.svg" align="middle" width="371.20466955pt" height="18.88772655pt"/></p>
Neither SMILES nor graph-based approaches guarantee this invariance. To the best of our knowledge SMILES-based approaches ignore this property altogether and graph-based approaches use Monte Carlo samples of permutations to try to average out the permutation choice (see for example, [GraphRNN You et. al. 2018]).

**Latent variable model**

Pretty much all of the above tasks use VAEs as the generative latent-variable model to generate graphs. We have been exploring the use of normalizing flows instead, which present the advantage of tractably yielding exact likelihoods instead of an evidence lower bound. To summarize, the graphical models look like:
<p align="center"><img alt="$$&#10;\text{VAE}: z \rightarrow x \quad\quad \text{Flow}: z \leftrightarrow x&#10;$$" src="svgs/b8691deab318e2fd1cbe740042d9acbe.svg" align="middle" width="215.88991380000002pt" height="11.4155283pt"/></p>
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

Let <img alt="$G=(V,E)$" src="svgs/56f934a7707b01a81435f67080a62e2f.svg" align="middle" width="78.51806489999998pt" height="24.65753399999998pt"/> denote a weighted undirected graph which we represent with a set of node features <img alt="$X \in \mathbb{R}^{n,m}$" src="svgs/596755c9f02e12a01425e4125ba9a393.svg" align="middle" width="70.56699044999999pt" height="22.648391699999998pt"/> and an adjacency matrix <img alt="$A \in \{0,1\}^{n,n}$" src="svgs/e909d92b706366ac5f3bbec0d2de7d15.svg" align="middle" width="92.75883704999998pt" height="24.65753399999998pt"/>, where <img alt="$n=|V|$" src="svgs/c7b684ce4dc092eef10985fbd8e50f44.svg" align="middle" width="54.15898619999999pt" height="24.65753399999998pt"/>. 

We suppose <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> is fixed, though later on this can easily be modeled as sampled from some parametric distribution. Another modeling possibility is to add discrete labels onto the edges by modeling <img alt="$A\in \{0,1,\dots,L\}^{n,n}$" src="svgs/426d48eed0a3bb26166d2c561d20a779.svg" align="middle" width="140.4754956pt" height="24.65753399999998pt"/> but we ignore that for now for the sake of simplicity.

We can factorize a probabilistic model of a graph into,
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p(X,A) &amp; = p(A)p(X|A).&#10;\end{align*}&#10;$$" src="svgs/504a4cd74dbfe1b6750fd02dedeb16a4.svg" align="middle" width="167.41439549999998pt" height="16.438356pt"/></p>

Ideally we want to learn a latent representation <img alt="$Z_1$" src="svgs/4f5bc204bf6a3d5abde8570c52d51cb6.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph structure <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/> and a latent representation <img alt="$Z_2$" src="svgs/3bd2daf9fde28292bb266114486cf619.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph nodes <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/> (such that structure and node features are disentangled). The graphical model looks like
<p align="center"><img alt="$$&#10;Z_1 \leftrightarrow A \rightarrow X \leftrightarrow Z_2.&#10;$$" src="svgs/e719b30552cd2618eb7439112f1c4c51.svg" align="middle" width="145.70737169999998pt" height="13.698590399999999pt"/></p>


We now describe how to model the latent representations <img alt="$Z_1,Z_2$" src="svgs/3aff0f15cf37c2947052a239490a68e2.svg" align="middle" width="43.67585144999999pt" height="22.465723500000017pt"/>.

---

#### Adjacency Matrix

The adjacency matrix is a discrete object. We can instead represent <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/> using de-quantized logits,
<p align="center"><img alt="$$&#10;\tilde{A} = \sigma^{-1}(A + \epsilon),&#10;$$" src="svgs/d82177d81da2319f2adc8a14dd6c3b5a.svg" align="middle" width="118.32180029999999pt" height="19.24333455pt"/></p>
where <img alt="$\epsilon$" src="svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg" align="middle" width="6.672392099999992pt" height="14.15524440000002pt"/> is appropriate noise. The flow process therefore looks like the below.
<p align="center"><img alt="$$&#10;Z_1 \sim N(0, \sigma^2I)\quad\quad \tilde{A} = f_\theta(Z_1)f_\theta(Z_1)^\top&#10;$$" src="svgs/376b7d84e8f8c209f873e539171719be.svg" align="middle" width="280.82763554999997pt" height="19.24333455pt"/></p>


Note that graphs have permutation symmetry; we let <img alt="$P$" src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg" align="middle" width="12.83677559999999pt" height="22.465723500000017pt"/> denote a permutation matrix of the same dimensionality as <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/>. Ideally we want
<p align="center"><img alt="$$&#10;p_\theta(A) = p_\theta(P^\top AP), \text{for all permutations }P.&#10;$$" src="svgs/566a68a2297d1e291b0b41ccc90435c1.svg" align="middle" width="311.79818174999997pt" height="18.88772655pt"/></p>
We use a *set transformer* to maintain permutation invariance in the function <img alt="$f_\theta$" src="svgs/953bed76e3f43e85c3d19bb59762fcc5.svg" align="middle" width="14.66328269999999pt" height="22.831056599999986pt"/>.
<p align="center"><img alt="$$&#10;f_\theta(Z_1) = [...]&#10;$$" src="svgs/9da6a38f45256b772e8c98dcccd92e27.svg" align="middle" width="91.61527815pt" height="16.438356pt"/></p>

---

Note that an alternative approach would be to follow [Kipf and Welling 2016], and use a variational auto-encoder instead of a discrete flow. 
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p_\theta(A|Z_1) &amp; = \sigma_\theta(Z_1 Z_1^\top)\\&#10;&#9;q_\phi(Z_1|A) &amp; = g_\phi(Z_1|A).&#10;\end{align*}&#10;$$" src="svgs/a7a7837ac18d47bc91775faa985b709d.svg" align="middle" width="157.96316579999998pt" height="44.1388464pt"/></p>


However, this does not exhibit permutation invariance and therefore did not work well in practice.

---

#### Node Features

Following [Liu et. al. 2019], we use bipartite (i.e. RealNVP) normalizing flows over a message-passing framework to describe a latent-variable model for <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/>. 
<p align="center"><img alt="$$&#10;Z_2 \sim N(0, \sigma^2I) \quad \quad X|A = f_\theta(Z_2;A).&#10;$$" src="svgs/f1985d89ead336699e3e95f2f2a28c65.svg" align="middle" width="267.36288974999997pt" height="18.312383099999998pt"/></p>
Using change of variables, the marginal likelihood is then given by
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p(f_\theta^{-1}(X;A))\left|\det \frac{\partial f_\theta^{-1}(X;A)}{\partial X}\right|.&#10;$$" src="svgs/07cc5334c537b8f0f1282e20ba5cc66b.svg" align="middle" width="313.02054135pt" height="41.4337374pt"/></p>


The message-passing framework yields <img alt="$p_\theta(X|A)$" src="svgs/348e6cdf5ca8da4523f0fadbf28870d8.svg" align="middle" width="60.29685749999999pt" height="24.65753399999998pt"/> that is permutation invariant; i.e.
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p_\theta(\Gamma X | \Gamma A), \text{for all permutations }\Gamma.&#10;$$" src="svgs/4e227812365269b1bd89785d531902bb.svg" align="middle" width="333.7902579pt" height="16.438356pt"/></p>
