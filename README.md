### Graph Embedding VAE

---

#### Summary

We build a generative model of graph structure by leveraging graph embeddings. We end up with a latent-variable model that is invariant to permutations of the node orderings and scalable to large graphs.

See our work at the NeurIPS 2019 Workshop on Graph Representation Learning:

Todo: link here. 

#### Code

Much of our evaluation code derives from Jiaxuan You's GraphRNN codebase [1]. Much of our attention code derives from Juho Lee's Set Transformer codebase [2].

#### References

[1] You, J., Ying, R., Ren, X., Hamilton, W., and Leskovec, J. (2018). GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models. In International Conference on Machine Learning, pp. 5708–5717.

[2] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

#### License

This code is provided under the MIT License.
