"""
transformer_encoder.py

Implementation of transformer encoder components including:
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Transformer encoder layer
  - Positional encoding
  - Full transformer encoder assembly

"""
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Attributes:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        q_linear (nn.Linear): Linear layer to project queries.
        k_linear (nn.Linear): Linear layer to project keys.
        v_linear (nn.Linear): Linear layer to project values.
        out_linear (nn.Linear): Linear layer for output projection.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the multi-head self-attention layers.

        Args:
            embed_dim (int): The total dimension of the input embeddings.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Perform forward pass for multi-head self-attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            output (Tensor): Contextualized output of shape (batch_size, seq_len, embed_dim).
            attention_weights (Tensor): Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = query.shape[0]

        # Linear projections and split into heads
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)

        # Concatenate heads and linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(context)
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Attributes:
        linear1 (nn.Linear): First linear transformation.
        linear2 (nn.Linear): Second linear transformation.
        relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self, embed_dim, ff_dim):
        """
        Initialize the feed-forward network.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            ff_dim (int): Inner-layer dimensionality of the feed-forward network.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Apply feed-forward transformations.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Output tensor of same shape as input.
        """
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward.

    Attributes:
        self_attn (MultiHeadSelfAttention): Self-attention sublayer.
        feed_forward (PositionwiseFeedForward): Feed-forward sublayer.
        norm1 (nn.LayerNorm): LayerNorm after attention.
        norm2 (nn.LayerNorm): LayerNorm after feed-forward.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize one encoder layer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout probability.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.

    Attributes:
        pe (Tensor): Positional encodings of shape (max_len, embed_dim).
    """
    def __init__(self, embed_dim, max_len=5000):
        """
        Create positional encodings.

        Args:
            embed_dim (int): Embedding size.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encodings to input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (seq_len, batch_size, embed_dim).

        Returns:
            Tensor: Positional encoded embeddings of same shape.
        """
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoder(nn.Module):
    """
    Full transformer encoder composed of multiple layers.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        layers (ModuleList): List of encoder layers.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, max_len=5000):
        """
        Initialize the transformer encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward hidden dimension.
            num_layers (int): Number of encoder layers.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Forward pass through the transformer encoder.

        Args:
            src (Tensor): Input token indices of shape (batch_size, seq_len).
            src_mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Encoder outputs of shape (batch_size, seq_len, embed_dim).
        """
        # Embedding lookup and scaling
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        # Add positional encoding
        src = self.pos_encoder(src)
        src = self.dropout(src)

        # Loop through encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

if __name__ == '__main__':
    # Example Usage
    vocab_size = 10000
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    max_len = 100

    # Create a dummy input sequence (batch_size, sequence_length)
    batch_size = 2
    sequence_length = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Instantiate the Transformer Encoder model
    model = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=max_len)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer Encoder model created and tested successfully!")


