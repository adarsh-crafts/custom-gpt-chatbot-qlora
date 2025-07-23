from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

# Head
class Head(nn.Module):
    '''one attention head in the self-attention'''

    def __init__(self, 
                 n_embed: int, 
                 head_size: int,
                 block_size: int,
                 dropout: float,
                 ) -> None:
        
        super().__init__()

        # creating the Q, K, and V layers; Learend during the optimization process
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias = False)

        # Attention mask
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)
        ))

        self.dropout = nn.Dropout(dropout)

    # defines the data flow
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # compute attention scores
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.50)

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )
        
        # softmax activation
        weights = F.softmax(weights, dim=-1)  # (B, T, T)

        # dropout 
        weights = self.dropout(weights)

        # weighted aggregation on Values
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out
    

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    ''' multiple heads of the self-attention in parallel'''

    def __init__(self, 
                 n_embed: int,
                 num_heads: int, 
                 head_size: int,
                 block_size: int,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim = -1)  # concatting outputs from all heads
        out = self.dropout(self.projection(out))  # projects back into n_embed -> performs regularization
        return out


# feed-Forward
class FeedForward(nn.Module):
    ''' simple Linear Layer followed by Non-Linear layer'''

    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # we make the space wider so the model can learn more complex patterns
            nn.ReLU(),  # activation function to introduce non-linearity (to cpature more complex relationships. ex: curvy shapes etc)
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

# Block
class Block(nn.Module):
    ''' Transformer Block: one full block'''

    def __init__(self, 
                 n_embed: int, 
                 n_head:int, 
                 block_size: int, 
                 dropout: float) -> None:
        super().__init__()

        head_size = n_embed // n_head

        error_message = f"n_embed {n_embed} must be divisible by n_head {n_head}"
        assert head_size * n_head == n_embed, error_message

        self.self_attention = MultiHeadAttention(
            n_embed=n_embed,
            num_heads=n_head,
            head_size=head_size, 
            block_size=block_size,
            dropout=dropout
            )
        self.feed_forward = FeedForward(n_embed, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embed)  # we use standard normalization
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # you can apply transformation before or after self attention
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# GPT Language Model
class GPTLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 n_embed: int,
                 n_head: int,
                 block_size: int,
                 n_layer: int,
                 dropout: float,
                 device: str,
                 ignore_index: int=-100
                 ) -> None:
        
        super().__init__()

        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.final_layer_norm = nn.LayerNorm(n_embed)
        self.final_linear_layer = nn.Linear(n_embed, vocab_size) 

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:        
        ''' 
            forward pass of the model.
            
            Args:
                input_tokens: Tensor of token indices of shape (batch_size, sequence_length)
                targets: optional tensor of target token indices of same shape as input_token

            Output:
                Tuple of (logits, loss) where logits has shape (batch_size, sequence_length, vocab_size)
                and loss is optional cross-entropy loss if targets are provided.
        '''
        B, T = input_tokens.shape

        token_embedding = self.token_embedding_table(input_tokens)  # (B, T, C)  # to convert (look at __init__())
        positional_embedding = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)

        x = token_embedding + positional_embedding  # (B, T, c)
        x = self.blocks(x)  # (B, T, c)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.final_linear_layer(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        '''
            Generate new tokens given a context.

            Args:
                input_tokens: starting token indices of shape (batch_size, sequence_length)
                max_new_tokens: no. of new tokens to generate
            
            Output:
                The generated tokens
        '''

        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)  # logits is (B, T, vocab-size)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim = 1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1  # (B, 1)
            )

        return input_tokens

    def advanced_generation(
            self,
            input_tokens: torch.Tensor,
            max_new_tokens: int,
            tempurature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None
            ) -> torch.Tensor:
        '''
            Generates new tokens from the model

            Args:
                input_tokens: the inital input tokens.
                max_new_tokens: max no. of tokens to generate
                temperature: controls randoms (higher->more random)
                top_k: limits generation to the top-k most likely tokens
                top_p: limits generation to tokens with cumulative probability <= top_p

            Returns:
                generated tokens
        '''
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :] / tempurature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('inf')

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)

        return input_tokens


if __name__ == '__main__':
    # example usage
    vocab_size = 16394
    embedding_size = 12
    number_of_heads = 8
    block_size = 1025
    number_of_blocks = 1
    dropout = 0.2
    head_size = embedding_size // number_of_heads
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embed=embedding_size,
        n_head = number_of_heads,
        block_size = block_size,
        n_layer = number_of_blocks,
        dropout = dropout,
        device = device
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Size: {model_size / 1e6:.2f}M parameters')

    print(f'Model created with {embedding_size=}, {number_of_heads=}, head_size{embedding_size//number_of_heads}')

    # dummy input
    input_tokens = torch.randint(0, vocab_size, (2, 50), device=device)

    # test forward pass
    # use input as target for testing shape
    logits, loss = model(input_tokens, targets= input_tokens)
    if loss is not None:
        print('Loss: ', loss.item())



    # test generation
    print('Generating...')

    # start generation from first 10 tokens
    generated_tokens = model.generate(input_tokens[:, :10], max_new_tokens=20)
    print('Generated tokens shape: ', generated_tokens.shape)
    print('Generated sequence example (first batch):\n', generated_tokens[0].tolist())


    # test advanced generation
    print('\n Advanced Generation (top_k=5, temp=0.8)...')
    generated_tokens_adv = model.advanced_generation(
        input_tokens[:, :10],
        max_new_tokens=20,
        temperature=20,
        tempurature=0.8,
        top_k=10
    )
    print('Generated tokens shape(adv): ', generated_tokens-generated_tokens_adv.shape)
    print('Generated sequence example (adv, fist batch):\n',
          generated_tokens_adv[0].tolist)