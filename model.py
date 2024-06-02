import torch
import numpy as np

# Positional Encoding Module
class PositionalEncoding(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_position=50):
        super(PositionalEncoding, self).__init__()
        # Linear transformation to map input dimension to output dimension
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        # Register a buffer for the position encoding table (not a model parameter)
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, out_dim))

    def _get_sinusoid_encoding_table(self, n_position, out_dim):
        '''Create sinusoid position encoding table'''
        def get_position_angle_vec(position):
            # Calculate position angle vector
            return [position / np.power(10000, 2 * (hid_j // 2) / out_dim) for hid_j in range(out_dim)]

        # Create the sinusoid table
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # Apply sin to even indices
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # Apply cos to odd indices

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # Apply linear transformation and add positional encoding
        x = self.linear(x)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

# Scaled Dot-Product Attention Module
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.shape[-1]
        # Calculate attention scores
        scores = torch.matmul(q / (d_k ** 0.5), k.transpose(2, 3)) #(N, n_head, T, T)
        if mask is not None:
            # Apply mask to the scores
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0)==0, -1e9)
        # Apply softmax to get attention weights
        scores = torch.nn.Softmax(dim=-1)(scores) #(N, n_head, T, T)
        # Calculate weighted sum of values
        output = torch.matmul(scores, v) #(N, n_head, T, out_dim)
        return output, scores

# Multi-Head Attention Module
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.out_dim = out_dim

        # Linear transformations for query, key, and value
        self.linear_q = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_k = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_v = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)

        self.scaled_dot_production_attention = ScaledDotProductAttention()
        # Linear transformation to combine the heads
        self.linear = torch.nn.Linear(in_features=n_head*out_dim, out_features=out_dim)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_kv = q.shape[0], q.shape[1], k.shape[1]

        # Apply linear transformations and split into heads
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.out_dim) #(N, T, n_head * out_dim)
        k = self.linear_k(k).view(batch_size, len_kv, self.n_head, self.out_dim)
        v = self.linear_v(v).view(batch_size, len_kv, self.n_head, self.out_dim)

        # Transpose for attention calculation
        q = q.transpose(1, 2) #(N, n_head, T, out_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply scaled dot-product attention
        output, scores = self.scaled_dot_production_attention(q, k, v, mask=mask)
        
        # Concatenate heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.linear(output) #(N, T, out_dim)
        return output, scores

# Position-wise Feed-Forward Network
class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim, out_features=in_dim)

    def forward(self, x):
        # Apply two linear transformations with ReLU in between
        x = self.linear_1(x)
        x = torch.nn.ReLU()(x)
        x = self.linear_2(x)
        return x

# Encoder Module
class Encoder(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(Encoder, self).__init__()
        # Positional encoding for the input
        self.position_enc = PositionalEncoding(in_dim, out_dim)
        # Multi-head attention layer
        self.multi_head_attention_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = torch.nn.LayerNorm(out_dim)
        # Position-wise feed-forward layer
        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_2 = torch.nn.LayerNorm(out_dim)
        self.scores_for_paint = None

    def forward(self, x):
        qkv = self.position_enc(x) #(N, T, 37) --> (N, T, 64) In the encoder, q, k, and v are the same

        # Apply multi-head attention
        residual = qkv # Residual connection
        outputs, scores = self.multi_head_attention_1(qkv, qkv, qkv) # Return scores for visualization
        self.scores_for_paint = scores.detach().cpu().numpy() # For visualization
        outputs = self.layer_norm_1_1(outputs + residual) # Add & Norm

        # Apply position-wise feed-forward network
        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_2(outputs + residual) # Add & Norm

        return outputs

# Function to create subsequent mask for the decoder
def get_subsequent_mask(seq):
    seq_len = seq.shape[1]
    ones = torch.ones((seq_len, seq_len), dtype=torch.int, device=seq.device)
    mask = 1 - torch.triu(ones, diagonal=1) # Mask out subsequent positions
    return mask

# Decoder Module
class Decoder(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(Decoder, self).__init__()
        # Positional encoding for the target sequence
        self.position_enc = PositionalEncoding(in_dim, out_dim)
        # Multi-head attention layers
        self.multi_head_attention_1_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = torch.nn.LayerNorm(out_dim)
        self.multi_head_attention_1_2 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_2 = torch.nn.LayerNorm(out_dim)
        # Position-wise feed-forward layer
        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_3 = torch.nn.LayerNorm(out_dim)
        self.scores_for_paint = None

    def forward(self, enc_outputs, target):
        qkv = self.position_enc(target) # In the encoder, q, k, and v are the same

        # Apply self-attention with masking for the target sequence
        residual = qkv # Residual connection
        outputs, scores = self.multi_head_attention_1_1(qkv, qkv, qkv, mask=get_subsequent_mask(target)) # Return scores for visualization
        outputs = self.layer_norm_1_1(outputs + residual) # Add & Norm

        # Apply attention with the encoder outputs
        residual = outputs
        outputs, scores = self.multi_head_attention_1_2(outputs, enc_outputs, enc_outputs)
        self.scores_for_paint = scores.detach().cpu().numpy() # For visualization
        outputs = self.layer_norm_1_2(outputs + residual)

        # Apply position-wise feed-forward network
        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_3(outputs + residual) # Add & Norm

        return outputs

# Transformer Module
class Transformer(torch.nn.Module):
    def __init__(self, n_head):
        super(Transformer, self).__init__()
        # Encoder and decoder
        self.encoder = Encoder(n_head, in_dim=37, out_dim=64) # 37 is the input dimension for encoder
        self.decoder = Decoder(n_head, in_dim=12, out_dim=64) # 12 is the input dimension for decoder
        # Linear transformation for the final output
        self.linear = torch.nn.Linear(in_features=64, out_features=12)

    def forward(self, x, y):
        enc_outputs = self.encoder(x)
        outputs = self.decoder(enc_outputs, y)
        outputs = self.linear(outputs)
        outputs = torch.nn.Softmax(dim=-1)(outputs)
        return outputs

    def size(self):
        size = sum([p.numel() for p in self.parameters()])
        print('%.2fKB' % (size * 4 / 1024))

if __name__ == '__main__':
    model = Transformer(n_head=4)
    model.size()

    batch_x = torch.randn(16, 10, 37) #(N, T, in_dim) - Input batch for the encoder
    batch_y = torch.randn(16, 7, 12) #(N, T, in_dim) - Input batch for the decoder

    pred = model(batch_x, batch_y)
    print(pred.shape) # Print shape of the output
    print(pred[0][0]) # Print the first prediction
