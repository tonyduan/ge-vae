# Normalizing Flows for Graphs Proposal

#### Idea

We follow the notation of [Grover et. al. 2019], repeated below. 

Let <img alt="$G=(V,E)$" src="svgs/56f934a7707b01a81435f67080a62e2f.svg" align="middle" width="78.51806489999998pt" height="24.65753399999998pt"/> denote a weighted undirected graph which we represent with a set of node features <img alt="$X \in \mathbb{R}^{n,m}$" src="svgs/596755c9f02e12a01425e4125ba9a393.svg" align="middle" width="70.56699044999999pt" height="22.648391699999998pt"/> and an adjacency matrix <img alt="$A \in \{0,1\}^{n,n}$" src="svgs/e909d92b706366ac5f3bbec0d2de7d15.svg" align="middle" width="92.75883704999998pt" height="24.65753399999998pt"/>, where <img alt="$n=|V|$" src="svgs/c7b684ce4dc092eef10985fbd8e50f44.svg" align="middle" width="54.15898619999999pt" height="24.65753399999998pt"/>. 

We suppose <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> is fixed, though this can easily be modeled as sampled from some parametric distribution. We can add discrete labels onto the edges by modeling <img alt="$A\in \{0,1,\dots,L\}^{n,n}$" src="svgs/426d48eed0a3bb26166d2c561d20a779.svg" align="middle" width="140.4754956pt" height="24.65753399999998pt"/> but ignore that for now for the sake of simplicity.

We can factorize a probabilistic model of the graphs into,
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p(X,A) &amp; = p(A)p(X|A).&#10;\end{align*}&#10;$$" src="svgs/504a4cd74dbfe1b6750fd02dedeb16a4.svg" align="middle" width="167.41439549999998pt" height="16.438356pt"/></p>

Ideally we want to learn a latent representation <img alt="$Z_1$" src="svgs/4f5bc204bf6a3d5abde8570c52d51cb6.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph structure <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/> and a latent representation <img alt="$Z_2$" src="svgs/3bd2daf9fde28292bb266114486cf619.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> for the graph nodes <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/> (such that structure and node features are disentangled). The graphical model looks like
<p align="center"><img alt="$$&#10;Z_1 \leftrightarrow A \rightarrow X \leftrightarrow Z_2.&#10;$$" src="svgs/e719b30552cd2618eb7439112f1c4c51.svg" align="middle" width="145.70737169999998pt" height="13.698590399999999pt"/></p>


We now describe how to model the latent representations <img alt="$Z_1,Z_2$" src="svgs/3aff0f15cf37c2947052a239490a68e2.svg" align="middle" width="43.67585144999999pt" height="22.465723500000017pt"/>.

---

#### Adjacency Matrix

Following [Tran et. al. 2019], we use discrete normalizing flows to describe a latent variable model for <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/>.  Ideally we want a flow to describe
<p align="center"><img alt="$$&#10;A = f_\theta(Z_1).&#10;$$" src="svgs/e718205e209d7ca6319eca15a488294a.svg" align="middle" width="85.67918205pt" height="16.438356pt"/></p>
Using change of variables, the marginal likelihood is given by (note no need for the Jacobian term)
<p align="center"><img alt="$$&#10;p_\theta(A) = p(f_\theta^{-1}(A)).&#10;$$" src="svgs/eeab56e01ea0acbf6349b3c1b97e7a39.svg" align="middle" width="140.94192255pt" height="19.40624895pt"/></p>


Note that graphs have permutation symmetry; we let <img alt="$\Gamma A$" src="svgs/a17070372c8cbc747d20be4164ee838b.svg" align="middle" width="22.60280219999999pt" height="22.465723500000017pt"/> denote a permutation operation over the rows of <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/>. Ideally we'd want
<p align="center"><img alt="$$&#10;p_\theta(A) = p_\theta(\Gamma A), \text{for all permutations }\Gamma.&#10;$$" src="svgs/3bdf538217b7c0c2e7dd3cad710da3b1.svg" align="middle" width="284.56647284999997pt" height="16.438356pt"/></p>
So what we need is for the flow to be permutation invariant over the rows. The answer is probably in [Bender et. al. 2019].

---

Note that an alternative approach would be to follow [Kipf and Welling 2016], and use a variational auto-encoder instead of a discrete flow. 
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;p_\theta(A|Z_1) &amp; = \sigma_\theta(Z_1 Z_1^\top)\\&#10;&#9;q_\phi(Z_1|A) &amp; = g_\phi(Z_1|A).&#10;\end{align*}&#10;$$" src="svgs/a7a7837ac18d47bc91775faa985b709d.svg" align="middle" width="157.96316579999998pt" height="44.1388464pt"/></p>


This would have the advantage of giving us continuous latent variables <img alt="$Z_1$" src="svgs/4f5bc204bf6a3d5abde8570c52d51cb6.svg" align="middle" width="17.77402769999999pt" height="22.465723500000017pt"/> that may be more interpretable, but has the disadvantage of introducing a variational approximation to the intractable posterior distribution <img alt="$q_\phi(Z_1|A)$" src="svgs/436c15524632dc2b3535403a59d85fb5.svg" align="middle" width="64.34073854999998pt" height="24.65753399999998pt"/>.

---

#### Node Features

Following [Liu et. al. 2019], we use bipartite (i.e. RealNVP) normalizing flows over a message-passing framework to describe a latent-variable model for <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/>. 
<p align="center"><img alt="$$&#10;X|A = f_\theta(Z_2;A).&#10;$$" src="svgs/530f267e254a46ac0a1c14eee13cd834.svg" align="middle" width="124.7887542pt" height="16.438356pt"/></p>
Using change of variables, the marginal likelihood is then given by
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p(f_\theta^{-1}(X;A))\left|\det \frac{\partial f_\theta^{-1}(X;A)}{\partial X}\right|.&#10;$$" src="svgs/07cc5334c537b8f0f1282e20ba5cc66b.svg" align="middle" width="313.02054135pt" height="41.4337374pt"/></p>


The message-passing framework yields <img alt="$p_\theta(X|A)$" src="svgs/348e6cdf5ca8da4523f0fadbf28870d8.svg" align="middle" width="60.29685749999999pt" height="24.65753399999998pt"/> that is permutation invariant; i.e.
<p align="center"><img alt="$$&#10;p_\theta(X|A) = p_\theta(\Gamma X | \Gamma A), \text{for all permutations }\Gamma.&#10;$$" src="svgs/4e227812365269b1bd89785d531902bb.svg" align="middle" width="333.7902579pt" height="16.438356pt"/></p>
