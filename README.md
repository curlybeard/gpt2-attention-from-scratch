# gpt2-attention-from-scratch
gpt 2 attention from scratch using pytorch, based on transformer architecture outlined in the 2017 attention is all you need paper: https://arxiv.org/abs/1706.03762

Trained on tinystories dataset (https://huggingface.co/datasets/roneneldan/TinyStories) to learn vocabularly and semantic representations of language using self supervised learning ie. masked auto regression

Causal self attention mechanism:
Query = What this token is looking for
Key = What information this token contains
Value = Actual content of the token that will be passed if Q and K match

How it works:
- initialize the query, key and value matrices from the input
- divide embedding size (768) by number of heads (12), each head is allocated 64 dimensions
- tranpose query and key matrices until sequence length is in row of query and column of key to create the 128 x 128 attention matrix
- divide the attentions scores by square root of the head dimension (sqrt(64) = 8) to keep gradients stable
- For causal masking, a lower triangluar matrix with negative infinity values for values above the main diagonal so masked future tokens get 0 attention. 
ie. (e^-inf = 0)
- Raw attention scores are converted via softmax into probabilities between 0 and 1 that sum to 1. 
- Multiply attention probabilities with the value matrix to compute final weighted representation of the tokens
- Concatenate 12 heads back into 768 dimesnions and pass through linear layer to add attention context before sending it to next block

Dependencies:
```
pip install torch datasets tiktoken
```

To train and save weights:

```
python train.py
```

To generate text from the trained model weights: 
```
python generate.py
```
