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

#### Path Forward

Our idea moving forward is to decompose a graph into its structure (adjacency matrix) and its node labels. We represent structure with a Laplacian embedding on each node, so that the embeddings on the nodes can be treated as a set (but without further permutation invariance within each embedding).  Further work plans to investigate the benefits of other embeddings, such as [Abu-El_haija et al. 2018]. Then we have a two-step process:
1. Use a set-invariant flow from Laplacian embeddings into Gaussian random vectors.
2. Use a separate decoder to predict pairwise edge probabilities from Laplacian embeddings.

One goal of our work is to scale up to generate large graphs after training on small graphs. 

Another possible idea (which we haven't pursued) to enforce permutation invariance is to permute each adjacency matrix (by degree of nodes, then left-ordering [Bloem-Reddy and Teh, 2019]) prior to training. Then the generation process can follow a sequential decoding procedure, enforcing for example the degree of each subsequent node.

---

### Proposal: normalizing flows for permutation-invariant graph generation

### Idea

We follow the notation of [Grover et al. 2019].

Let <img alt="$G=(V,E)$" src="svgs/56f934a7707b01a81435f67080a62e2f.svg" align="middle" width="78.51806489999998pt" height="24.65753399999998pt"/> denote a weighted undirected graph which we represent with a set of node features <img alt="$X \in \mathbb{R}^{n,m}$" src="svgs/596755c9f02e12a01425e4125ba9a393.svg" align="middle" width="70.56699044999999pt" height="22.648391699999998pt"/> and an adjacency matrix <img alt="$A \in \{0,1\}^{n,n}$" src="svgs/e909d92b706366ac5f3bbec0d2de7d15.svg" align="middle" width="92.75883704999998pt" height="24.65753399999998pt"/>, where <img alt="$n=|V|$" src="svgs/c7b684ce4dc092eef10985fbd8e50f44.svg" align="middle" width="54.15898619999999pt" height="24.65753399999998pt"/>. 

We suppose <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> is fixed, though later on this can easily be modeled as sampled from some parametric distribution. Another modeling possibility is to add discrete labels onto the edges by modeling <img alt="$A\in \{0,1,\dots,L\}^{n,n}$" src="svgs/426d48eed0a3bb26166d2c561d20a779.svg" align="middle" width="140.4754956pt" height="24.65753399999998pt"/> but we ignore that for now for the sake of simplicity.

We can factorize a probabilistic model of a graph into,
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p(X,A) &amp; = p(A)p(X|A).&#10;\end{align*}&#10;$$" src="svgs/504a4cd74dbfe1b6750fd02dedeb16a4.svg" align="middle" width="167.41439549999998pt" height="16.438356pt"/></p>

That is, we want to learn a latent representation <img alt="$Z_1$" src="svgs/4f5bc204bf6a3d5abde8570c52d51cb6.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph structure <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/> and a latent representation <img alt="$Z_2$" src="svgs/3bd2daf9fde28292bb266114486cf619.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph nodes <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/> (such that structure and node features are disentangled). The graphical model looks like
<p align="center"><img alt="$$&#10;Z_1 \leftrightarrow A \rightarrow X \leftrightarrow Z_2.&#10;$$" src="svgs/e719b30552cd2618eb7439112f1c4c51.svg" align="middle" width="145.70737169999998pt" height="13.698590399999999pt"/></p>


We now describe how to model the latent representations <img alt="$Z_1,Z_2$" src="svgs/3aff0f15cf37c2947052a239490a68e2.svg" align="middle" width="43.67585144999999pt" height="22.465723500000017pt"/>.

---

#### Adjacency Matrix

The adjacency matrix is a discrete object, so it is not immediately straightforward how to apply normalizing flows. We choose to use *graph embeddings* to encode the structure of the graph. We can apply a deterministic embedding model to map each
<p align="center"><img alt="$$&#10;A \rightarrow E,&#10;$$" src="svgs/7c01960697f225163ec7631ca3fa7235.svg" align="middle" width="55.54779944999999pt" height="14.42921205pt"/></p>
where the permutation invariance on <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" align="middle" width="13.08219659999999pt" height="22.465723500000017pt"/> must satisfy
<p align="center"><img alt="$$&#10;p_\theta(PE) = p_\theta(E), \text{for all permutation matrices }P.&#10;$$" src="svgs/0fe00d824faef1e1c16e6ba175670d0c.svg" align="middle" width="348.7787391pt" height="16.438356pt"/></p>
The set transformer is able to satisfy this invariance. Therefore we use a normalizing flow for
<p align="center"><img alt="$$&#10;Z_1 \leftrightarrow E, \quad\quad Z_1 \sim N(0, \sigma^2 I).&#10;$$" src="svgs/68645ce6e87c4678982f1196c16e4994.svg" align="middle" width="211.69496039999996pt" height="18.312383099999998pt"/></p>
One challenge we observed is that there exists significant multi-modality in <img alt="$E$" src="svgs/84df98c65d88c6adf15d4645ffa25e47.svg" align="middle" width="13.08219659999999pt" height="22.465723500000017pt"/>. For example, in a two-community graph there will be one mode for each community. Naive use of a normalizing flow was unable to deal with this issue; however, the neural spline flow [Durkan et al. 2019] has seen success at this task.

This gives us a permutation-invariant *likelihood* model. However in order to generate graphs we need some way to decode 
<p align="center"><img alt="$$&#10;E \rightarrow A.&#10;$$" src="svgs/6c720665e41643d0d8c3a553d3ff8e1a.svg" align="middle" width="55.54779944999999pt" height="11.232861749999998pt"/></p>
The most precise approach is to store a database of mappings <p align="center"><img alt="$$E \leftrightarrow A$$" src="svgs/22e24b2ca54e9e069d0e413a3e8dc268.svg" align="middle" width="50.981576249999996pt" height="11.232861749999998pt"/></p> and perform a lookup; however this would require <img alt="$O(2^{n^2})$" src="svgs/d82b67c5881fbd29811a4a071b16ff1c.svg" align="middle" width="49.36361924999999pt" height="32.44583099999998pt"/> entries where <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> is the number of nodes in the graph. This does not scale, so instead we train a decoder
<p align="center"><img alt="$$&#10;p_\phi(A_{i,j} = 1 | E_i,E_j)&#10;$$" src="svgs/1fe03e3e159b88ac54f44a5ab76b5e90.svg" align="middle" width="136.26980565pt" height="17.031940199999998pt"/></p>
in order to generate the adjacency matrix <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/>. Unfortunately this adds some stochasticity to the generation layer, but there doesn't appear to be a better way to do this for now.

---

As a baseline, we consider the graph variational auto-encoder of [Kipf and Welling 2016], instead of a flow. Their model factorizes
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p_\theta(A|Z_1) &amp; = \sigma_\theta(Z_1 Z_1^\top)\\&#10;&#9;q_\phi(Z_1|A) &amp; = g_\phi(Z_1|A).&#10;\end{align*}&#10;$$" src="svgs/a7a7837ac18d47bc91775faa985b709d.svg" align="middle" width="157.96316579999998pt" height="44.1388464pt"/></p>


However, this model does not exhibit permutation invariance and we suspect it will not work as well on downstream tasks.

---

#### Node Features

Following [Liu et. al. 2019], we use bipartite (i.e. RealNVP) normalizing flows over a message-passing model to describe a latent-variable model for <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/>. 
<p align="center"><img alt="$$&#10;Z_2 \sim N(0, \sigma^2I) \quad \quad X|A = f_\theta(Z_2;A).&#10;$$" src="svgs/f1985d89ead336699e3e95f2f2a28c65.svg" align="middle" width="267.36288974999997pt" height="18.312383099999998pt"/></p>
Using change of variables, the marginal likelihood is then given by
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p(f_\theta^{-1}(X;A))\left|\det \frac{\partial f_\theta^{-1}(X;A)}{\partial X}\right|.&#10;$$" src="svgs/07cc5334c537b8f0f1282e20ba5cc66b.svg" align="middle" width="313.02054135pt" height="41.4337374pt"/></p>


The message-passing framework yields <img alt="$p_\theta(X|A)$" src="svgs/348e6cdf5ca8da4523f0fadbf28870d8.svg" align="middle" width="60.29685749999999pt" height="24.65753399999998pt"/> that is permutation invariant; i.e.
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p_\theta(\Gamma X | \Gamma A), \text{for all permutations }\Gamma.&#10;$$" src="svgs/4e227812365269b1bd89785d531902bb.svg" align="middle" width="333.7902579pt" height="16.438356pt"/></p>
