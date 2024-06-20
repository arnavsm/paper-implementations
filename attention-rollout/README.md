```This is an implementation of attention rollout for vision transformers```

Attention rollout is a method for visualizing and interpreting the attention mechanism in Transformer models. It addresses the issue that attention weights in deeper layers of Transformer models become increasingly mixed and less interpretable.

The key idea behind attention rollout is to recursively compute the token attentions in each layer of the Transformer by multiplying the attention matrices across layers. This allows the attention signal to be propagated through the entire network, rather than just being visible in the first few layers.

Specifically, the attention rollout algorithm works as follows:

1. Start with the attention matrix from the first layer.
2. For each subsequent layer, multiply the current layer's attention matrix with the previous layer's attention rollout matrix.
3. Normalize the rows of the final attention rollout matrix to ensure the total attention flow sums to 1.

This recursive computation allows the method to capture how information flows through the self-attention layers of the Transformer.

Attention rollout has been shown to produce more interpretable and meaningful visualizations of attention, especially for deeper layers of Transformer models, compared to just looking at the raw attention weights. It provides a way to quantify how information propagates through the self-attention mechanism.

Citations:
1. [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)
2. [Official Code](https://github.com/samiraabnar/attention_flow)
3. [Blogpost - Exploring Explainability for Vision Transformers](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)
4. Thank you to [@hankyul2](https://github.com/hankyul2) with [this](https://github.com/huggingface/pytorch-image-models/discussions/1232)