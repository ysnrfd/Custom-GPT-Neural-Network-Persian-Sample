"""

LICENSE:



Copyright 2025 ysnrfd



Timestamp: 2025-08-12



Permission is hereby granted, free of charge, to any person obtaining a copy

of this software and associated documentation files (the "Software"), to use,

copy, modify, and distribute the Software, subject to the following conditions:



1. The copyright notice, this permission notice, and all attribution information

   regarding the original author (ysnrfd) must be preserved in their entirety

   and must not be removed, altered, or obscured in any copies or derivative works.



2. Any modifications or derivative works must be clearly documented in a "CHANGELOG" or

   "NOTICE" file included with the Software. This documentation must include a detailed

   description of the changes made, the date of the modification, and the identity of

   the modifier.



3. The Software is provided "as is", without warranty of any kind, express or implied.

   The author shall not be liable for any damages arising from use of the Software.



4. Any attempt to remove or alter the original attribution or copyright information

   constitutes a violation of this license and may result in legal action.



"""



import math

import numpy as np

import pickle

import os

import time

from typing import List, Tuple, Dict, Any, Optional, Union, Set

import warnings



# --- Global Configuration and Utility Functions ---

DEFAULT_DTYPE = np.float32

EPS = 1e-6



def softmax(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:

    """Stable softmax implementation."""

    x = x - np.max(x, axis=axis, keepdims=True)

    e = np.exp(x)

    return e / (np.sum(e, axis=axis, keepdims=True) + eps)



def gelu(x: np.ndarray) -> np.ndarray:

    """Gated Linear Unit (GELU) approximation."""

    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))



def gelu_grad(x: np.ndarray) -> np.ndarray:

    """Gradient of the GELU approximation."""

    # This is a complex derivative, calculated based on the approximation formula

    tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))

    sech2 = 1.0 - tanh_term**2

    inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)

    return 0.5 * (1.0 + tanh_term) + 0.5 * x * sech2 * inner_grad



def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:

    """Root Mean Square Normalization (RMSNorm) utility."""

    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    return weight * (x / rms)



# --- Tokenizer (BPE) Implementation ---



class BPETokenizer:

    """A simple, single-file implementation of a BPE tokenizer."""

    def __init__(self):

        self.vocab: List[str] = []

        self.w2i: Dict[str, int] = {}

        self.i2w: Dict[int, str] = {}

        self.merges: List[Tuple[str, str]] = []

        self.cache: Dict[str, List[str]] = {}

        self.special_tokens: List[str] = ['<pad>', '<unk>', '<bos>', '<eos>']



    @staticmethod

    def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:

        """Helper to get all adjacent pairs in a sequence of tokens."""

        return set(zip(word, word[1:]))



    @staticmethod

    def bytes_to_unicode() -> Dict[int, str]:

        """Maps bytes to a custom set of unicode characters to handle all byte values."""

        bs = list(range(ord("!"), ord("~") + 1)) + \

             list(range(ord("¡"), ord("¬") + 1)) + \

             list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]

        n = 0

        for b in range(2**8):

            if b not in bs:

                bs.append(b)

                cs.append(2**8 + n)

                n += 1

        cs = [chr(n) for n in cs]

        return dict(zip(bs, cs))



    def preprocess(self, text: str) -> str:

        """Converts text to bytes and maps bytes to special unicode for BPE processing."""

        byte_encoder = self.bytes_to_unicode()

        text_bytes = text.encode("utf-8")

        return "".join([byte_encoder[b] for b in text_bytes])



    def build_from_text(self, texts: List[str], vocab_size: int = 500, min_freq: int = 2):

        """Builds the BPE vocabulary and merge rules from a list of texts."""

        preprocessed = [self.preprocess(text) for text in texts]

        # 1. Initialize with basic characters

        char_freq = {}

        for text in preprocessed:

            for char in text:

                char_freq[char] = char_freq.get(char, 0) + 1

        

        # Filter initial vocab by min_freq for a more robust start if needed, 

        # but here we prioritize special tokens and all chars for completeness.

        self.vocab = self.special_tokens + sorted(char_freq.keys())

        self.w2i = {w: i for i, w in enumerate(self.vocab)}

        self.i2w = {i: w for w, i in self.w2i.items()}



        # 2. Prepare initial words for merging

        word_list = []

        for text in preprocessed:

            word_list.append(list(text)) # Start with characters as tokens

            

        # 3. Perform BPE merging

        num_merges = vocab_size - len(self.vocab)

        for _ in range(num_merges):

            if len(self.vocab) >= vocab_size: break



            pair_counts: Dict[Tuple[str, str], int] = {}

            for tokens in word_list:

                pairs = self.get_pairs(tuple(tokens))

                for pair in pairs:

                    pair_counts[pair] = pair_counts.get(pair, 0) + 1



            if not pair_counts:

                break



            best_pair = max(pair_counts, key=pair_counts.get)

            if pair_counts[best_pair] < min_freq: 

                break # Stop if best pair is too infrequent



            new_token = ''.join(best_pair)

            

            if new_token not in self.w2i:

                self.vocab.append(new_token)

                self.w2i[new_token] = len(self.vocab) - 1

                self.i2w[len(self.vocab) - 1] = new_token

                self.merges.append(best_pair)

            

            # Apply the new merge rule to all words

            new_word_list = []

            for tokens in word_list:

                new_tokens = []

                i = 0

                while i < len(tokens):

                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:

                        new_tokens.append(new_token)

                        i += 2

                    else:

                        new_tokens.append(tokens[i])

                        i += 1

                new_word_list.append(new_tokens)

            word_list = new_word_list

    

    def encode(self, text: str, max_len: int = None, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:

        """Tokenizes text using the learned merge rules."""

        text_to_process = text

        if add_bos:

            text_to_process = self.special_tokens[2] + text_to_process

        if add_eos:

            text_to_process = text_to_process + self.special_tokens[3]

        

        # If it's a known special token sequence, use it directly

        if text_to_process in self.w2i:

            ids = [self.w2i[text_to_process]]

        else:

            text_preprocessed = self.preprocess(text)

            tokens = list(text_preprocessed)

            

            for pair in self.merges:

                new_tokens = []

                i = 0

                while i < len(tokens):

                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:

                        new_tokens.append(pair[0] + pair[1])

                        i += 2

                    else:

                        new_tokens.append(tokens[i])

                        i += 1

                tokens = new_tokens

            

            ids = [self.w2i.get(t, self.w2i['<unk>']) for t in tokens]

            

        if max_len is not None and len(ids) > max_len:

            ids = ids[:max_len]

        if max_len is not None and len(ids) < max_len:

            ids = ids + [self.w2i['<pad>']] * (max_len - len(ids))

            

        return np.array(ids, dtype=np.int32)



    def decode(self, ids: Union[np.ndarray, List[int]]) -> str:

        """Converts token IDs back to human-readable text."""

        tokens = [self.i2w.get(int(i), '<unk>') for i in ids]

        text = ''.join(tokens)

        for token in self.special_tokens:

            text = text.replace(token, '')

        byte_decoder = {v: k for k, v in self.bytes_to_unicode().items()}

        text_bytes = bytearray([byte_decoder[c] for c in text if c in byte_decoder])

        return text_bytes.decode('utf-8', errors='replace')



# --- Core Neural Network Components ---



class Embedding:

    def __init__(self, vocab_size: int, d_model: int, dtype=DEFAULT_DTYPE):

        self.vocab_size = vocab_size

        self.d_model = d_model

        self.dtype = dtype

        scale = 1.0 / np.sqrt(d_model)

        self.W = np.random.normal(0, scale, (vocab_size, d_model)).astype(dtype)

        self.grad_W = np.zeros_like(self.W)



    def forward(self, idx: np.ndarray) -> np.ndarray:

        """Maps token indices to dense vectors."""

        return self.W[idx]



    def backward(self, idx: np.ndarray, grad: np.ndarray):

        """Accumulates gradients for the embedding matrix W."""

        # np.add.at is used for efficient scatter-add operation

        np.add.at(self.grad_W, idx, grad)



class PositionalEmbedding:

    def __init__(self, max_len: int, d_model: int, use_rotary: bool = False, dtype=DEFAULT_DTYPE):

        self.max_len = max_len

        self.d_model = d_model

        self.use_rotary = use_rotary

        self.dtype = dtype

        if not use_rotary:

            # Standard Sinusoidal Positional Encoding (Absolute)

            self.W = np.zeros((max_len, d_model), dtype=dtype)

            for pos in range(max_len):

                for i in range(0, d_model, 2):

                    self.W[pos, i] = math.sin(pos / (10000 ** (i / d_model)))

                    if i + 1 < d_model:

                        self.W[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

            self.grad_W = np.zeros_like(self.W)

        else:

            # Rotary Positional Embedding (RoPE) - Frequencies only

            self.rotary_freqs = self._create_rotary_frequencies()

            # No self.W or self.grad_W needed for RoPE, as it's applied dynamically



    def _create_rotary_frequencies(self) -> np.ndarray:

        """Calculates the inverse frequencies for RoPE."""

        inv_freq = 1.0 / (10000 ** (np.arange(0, self.d_model, 2, dtype=self.dtype) / self.d_model))

        return inv_freq



    @staticmethod

    def apply_rotary_pos_emb(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:

        """Applies RoPE to a tensor (e.g., Q or K)."""

        # x shape: (B, H, S, D_head) or (B, S, D)

        seq_len = x.shape[-2] # Assume sequence length is the second to last dimension

        t = np.arange(seq_len, dtype=x.dtype)

        freqs = np.outer(t, freqs) # (S, D_half)

        

        cos = np.cos(freqs)[np.newaxis, np.newaxis, :, :] # (1, 1, S, D_half)

        sin = np.sin(freqs)[np.newaxis, np.newaxis, :, :] # (1, 1, S, D_half)

        

        # Split x into x1 (even indices) and x2 (odd indices)

        x1 = x[..., 0::2]

        x2 = x[..., 1::2]

        

        # Perform the rotation

        x_rotated1 = x1 * cos - x2 * sin

        x_rotated2 = x1 * sin + x2 * cos

        

        # Combine back

        x_rotated = np.zeros_like(x)

        x_rotated[..., 0::2] = x_rotated1

        x_rotated[..., 1::2] = x_rotated2

        

        return x_rotated



    def forward(self, seq_len: int) -> Optional[np.ndarray]:

        """Returns positional embeddings if not using RoPE."""

        if not self.use_rotary:

            # W shape is (max_len, d_model), need (1, seq_len, d_model)

            return self.W[:seq_len][np.newaxis, :, :]

        return None # RoPE is applied inside MHA



    def backward(self, seq_len: int, grad: np.ndarray):

        """Accumulates gradients for absolute positional embeddings."""

        if not self.use_rotary:

            # Sum gradient across batch and sequence dimension, then scatter-add to W

            np.add.at(self.grad_W, np.arange(seq_len), np.sum(grad, axis=0))



class LayerNorm:

    def __init__(self, d_model: int, eps: float = EPS, rms_norm: bool = False, dtype=DEFAULT_DTYPE):

        self.d_model = d_model

        self.eps = eps

        self.rms_norm = rms_norm

        self.dtype = dtype

        if not rms_norm:

            # Standard LayerNorm (gamma * (x - mean) / std + beta)

            self.gamma = np.ones((1, 1, d_model), dtype=dtype)

            self.beta = np.zeros((1, 1, d_model), dtype=dtype)

            self.grad_gamma = np.zeros_like(self.gamma)

            self.grad_beta = np.zeros_like(self.beta)

            self.weight = None # For compatibility in the model structure

            self.grad_weight = None

        else:

            # RMSNorm (weight * x / rms)

            self.weight = np.ones((1, 1, d_model), dtype=dtype)

            self.grad_weight = np.zeros_like(self.weight)

            self.gamma = None

            self.beta = None

            self.grad_gamma = None

            self.grad_beta = None

        

        # Cache for backward pass

        self.x = None

        self.x_norm = None

        self.var_or_rms = None # Stores variance for LN or RMS for RMSN



    def forward(self, x: np.ndarray) -> np.ndarray:

        self.x = x

        if self.rms_norm:

            rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)

            self.var_or_rms = rms # Cache RMS

            self.x_norm = x / rms

            return self.weight * self.x_norm

        else:

            mean = np.mean(x, axis=-1, keepdims=True)

            var = np.var(x, axis=-1, keepdims=True)

            self.var_or_rms = var # Cache variance

            std = np.sqrt(var + self.eps)

            self.x_norm = (x - mean) / std

            return self.gamma * self.x_norm + self.beta



    def backward(self, grad: np.ndarray) -> np.ndarray:

        if self.rms_norm:

            # RMSNorm Backward Pass (Simplified)

            grad_x_norm = grad * self.weight

            self.grad_weight = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)

            

            rms = self.var_or_rms

            x_norm = self.x_norm

            

            # d(1/rms) = -1/rms^2 * d(rms)

            # d(rms) = 1/(2*rms) * d(rms^2)

            # d(rms^2) = 1/D * 2 * x * dx

            

            # Calculate the derivative of 1/rms w.r.t x (d_inv_rms)

            d_inv_rms = -(x * np.sum(grad_x_norm * x_norm, axis=-1, keepdims=True)) / (self.d_model * rms**3)

            

            # Full derivative dx = grad_x_norm * (1/rms) + grad_x_norm * x * d(1/rms)/dx

            dx = grad_x_norm / rms + d_inv_rms

            return dx

        else:

            # Standard LayerNorm Backward Pass

            b, s, d = grad.shape

            self.grad_gamma = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)

            self.grad_beta = np.sum(grad, axis=(0, 1), keepdims=True)



            dx_norm = grad * self.gamma

            var_eps = self.var_or_rms + self.eps

            

            # 1. Backprop through normalization (x - mean) / std

            # dL/dx = 1/std * (dL/dy_norm - mean(dL/dy_norm) - x_norm * mean(dL/dy_norm * x_norm))

            dx = (1. / np.sqrt(var_eps)) * (dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) - 

                        self.x_norm * np.mean(dx_norm * self.x_norm, axis=-1, keepdims=True))

            return dx



class FeedForward:

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, dtype=DEFAULT_DTYPE):

        self.d_model = d_model

        self.d_ff = d_ff

        self.dropout = dropout

        self.dtype = dtype

        

        scale_in = 1.0 / np.sqrt(d_model)

        scale_out = 1.0 / np.sqrt(d_ff)

        

        # Parameters

        self.W1 = np.random.normal(0, scale_in, (d_model, d_ff)).astype(dtype)

        self.b1 = np.zeros((1, 1, d_ff), dtype=dtype)

        self.W2 = np.random.normal(0, scale_out, (d_ff, d_model)).astype(dtype)

        self.b2 = np.zeros((1, 1, d_model), dtype=dtype)

        

        # Gradients

        self.grad_W1 = np.zeros_like(self.W1)

        self.grad_b1 = np.zeros_like(self.b1)

        self.grad_W2 = np.zeros_like(self.W2)

        self.grad_b2 = np.zeros_like(self.b2)

        

        # Cache

        self.x = None

        self.hidden = None

        self.hidden_act = None

        self.dropout_mask1 = None

        self.dropout_mask2 = None



    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:

        self.x = x

        # Layer 1: Linear + GELU

        self.hidden = x @ self.W1 + self.b1

        self.hidden_act = gelu(self.hidden)

        

        # Dropout 1

        if training and self.dropout > 0:

            self.dropout_mask1 = (np.random.rand(*self.hidden_act.shape) > self.dropout) / (1 - self.dropout)

            self.hidden_act = self.hidden_act * self.dropout_mask1

        else:

            self.dropout_mask1 = None

            

        # Layer 2: Linear

        out = self.hidden_act @ self.W2 + self.b2

        

        # Dropout 2 (Often applied after residual connection in practice, but placed here per user structure)

        if training and self.dropout > 0:

            self.dropout_mask2 = (np.random.rand(*out.shape) > self.dropout) / (1 - self.dropout)

            out = out * self.dropout_mask2

        else:

            self.dropout_mask2 = None

            

        return out



    def backward(self, grad: np.ndarray) -> np.ndarray:

        b, s, d = grad.shape

        

        # 1. Backprop through Dropout 2 (if applied)

        if self.dropout_mask2 is not None:

            grad = grad * self.dropout_mask2

            

        # 2. Backprop through W2 + b2

        self.grad_W2 = (self.hidden_act.reshape(-1, self.d_ff).T @ grad.reshape(-1, d))

        self.grad_b2 = np.sum(grad, axis=(0, 1), keepdims=True)

        dhidden_act = grad @ self.W2.T

        

        # 3. Backprop through Dropout 1 (if applied)

        if self.dropout_mask1 is not None:

            dhidden_act = dhidden_act * self.dropout_mask1

            

        # 4. Backprop through GELU activation

        dhidden = dhidden_act * gelu_grad(self.hidden)

        

        # 5. Backprop through W1 + b1

        self.grad_W1 = (self.x.reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_ff))

        self.grad_b1 = np.sum(dhidden, axis=(0, 1), keepdims=True)

        

        # 6. Backprop to input x

        dx = dhidden @ self.W1.T

        return dx



class MultiHeadSelfAttention:

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_rotary: bool = False, dtype=DEFAULT_DTYPE):

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model

        self.num_heads = num_heads

        self.head_dim = d_model // num_heads

        self.dropout = dropout

        self.use_rotary = use_rotary

        self.dtype = dtype

        

        # Parameters (W_q, W_k, W_v, W_o are the same size: (d_model, d_model))

        scale = 1.0 / np.sqrt(d_model)

        self.W_q = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)

        self.W_k = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)

        self.W_v = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)

        self.W_o = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)

        

        # Gradients

        self.grad_W_q = np.zeros_like(self.W_q)

        self.grad_W_k = np.zeros_like(self.W_k)

        self.grad_W_v = np.zeros_like(self.W_v)

        self.grad_W_o = np.zeros_like(self.W_o)

        

        # Cache

        self.cache: Dict[str, Any] = {}

        self.dropout_mask = None



    def split_heads(self, x: np.ndarray) -> np.ndarray:

        """Splits the last dimension into (num_heads, head_dim) and transposes."""

        b, s, d = x.shape

        x = x.reshape(b, s, self.num_heads, self.head_dim)

        return np.transpose(x, (0, 2, 1, 3)) # Shape (B, H, S, D_h)



    def combine_heads(self, x: np.ndarray) -> np.ndarray:

        """Transposes back and combines (num_heads, head_dim) into d_model."""

        x = np.transpose(x, (0, 2, 1, 3)) # Shape (B, S, H, D_h)

        b, s, h, hd = x.shape

        return x.reshape(b, s, h * hd) # Shape (B, S, D)



    def causal_mask(self, seq_len: int) -> np.ndarray:

        """Creates an upper-triangular mask for attention (GPT-style)."""

        return np.tril(np.ones((seq_len, seq_len), dtype=bool))



    def forward(self, x: np.ndarray, pos_freqs: Optional[np.ndarray], training: bool = True) -> np.ndarray:

        b, s, d = x.shape

        # 1. Project Q, K, V

        Q = x @ self.W_q

        K = x @ self.W_k

        V = x @ self.W_v

        

        # 2. Split into multiple heads

        Qh = self.split_heads(Q)

        Kh = self.split_heads(K)

        Vh = self.split_heads(V)

        

        # 3. Apply Rotary Positional Embedding (if enabled)

        if self.use_rotary and pos_freqs is not None:

            # We reuse the static method and pass the frequencies from PositionalEmbedding

            Qh = PositionalEmbedding.apply_rotary_pos_emb(Qh, pos_freqs)

            Kh = PositionalEmbedding.apply_rotary_pos_emb(Kh, pos_freqs)



        # 4. Scaled Dot-Product Attention

        dk = self.head_dim

        scores = Qh @ np.swapaxes(Kh, -1, -2) / np.sqrt(dk) # (B, H, S, S)

        

        # 5. Apply Causal Mask

        mask = self.causal_mask(s)[np.newaxis, np.newaxis, :, :]

        scores = np.where(mask, scores, -1e10) # Use a large negative number instead of -np.inf



        # 6. Softmax

        attn = softmax(scores, axis=-1)

        

        # 7. Dropout

        if training and self.dropout > 0:

            self.dropout_mask = (np.random.rand(*attn.shape) > self.dropout) / (1 - self.dropout)

            attn = attn * self.dropout_mask

        else:

            self.dropout_mask = None

            

        # 8. Apply attention to V

        attn_out = attn @ Vh # (B, H, S, D_h)



        # 9. Combine heads and output projection

        out_combined = self.combine_heads(attn_out)

        out = out_combined @ self.W_o



        # Cache necessary values for backprop

        self.cache = {

            'x': x, 'Q': Q, 'K': K, 'V': V,

            'Qh': Qh, 'Kh': Kh, 'Vh': Vh,

            'scores': scores, 'attn': attn, 'attn_out': attn_out,

            'out_combined': out_combined

        }

        return out



    def backward(self, grad_out: np.ndarray) -> np.ndarray:

        x = self.cache['x']

        Qh = self.cache['Qh']

        Kh = self.cache['Kh']

        Vh = self.cache['Vh']

        attn = self.cache['attn']

        attn_out = self.cache['attn_out']

        out_combined = self.cache['out_combined']

        b, s, d = grad_out.shape

        dk = self.head_dim



        # 1. Backprop through W_o projection

        self.grad_W_o = out_combined.reshape(-1, d).T @ grad_out.reshape(-1, d)

        d_out_combined = grad_out @ self.W_o.T # (B, S, D)

        

        # 2. Backprop through combine_heads

        d_attn_out = d_out_combined.reshape(b, s, self.num_heads, self.head_dim) # (B, S, H, D_h)

        d_attn_out = np.transpose(d_attn_out, (0, 2, 1, 3)) # (B, H, S, D_h)



        # 3. Backprop through Vh and Attention Matrix (attn @ Vh)

        dVh = np.matmul(np.swapaxes(attn, -1, -2), d_attn_out) # (B, H, S, D_h)

        dattn = np.matmul(d_attn_out, np.swapaxes(Vh, -1, -2)) # (B, H, S, S)



        # 4. Backprop through Dropout (if applied)

        if self.dropout_mask is not None:

            dattn = dattn * self.dropout_mask



        # 5. Backprop through Softmax and Mask

        sft = attn

        # Sum of (dattn * sft) over the last axis

        sum_d = np.sum(dattn * sft, axis=-1, keepdims=True)

        dscores = sft * (dattn - sum_d)

        

        # Re-apply the mask to zero-out gradients for masked positions

        mask = self.causal_mask(s)[np.newaxis, np.newaxis, :, :]

        dscores = np.where(mask, dscores, 0.0)



        # 6. Backprop through dot product (scores = Qh @ Kh.T / sqrt(dk))

        dQh = np.matmul(dscores, Kh) / np.sqrt(dk) # (B, H, S, D_h)

        dKh = np.matmul(np.swapaxes(dscores, -1, -2), Qh) / np.sqrt(dk) # (B, H, S, D_h)



        # NOTE: RoPE gradients are implicitly captured here as the operation 

        # is linear *before* the non-linear attention and only depends on Q/K.

        # Backprop through RoPE would require complex, explicit rotation matrix derivatives,

        # which is usually simplified by just passing the gradient back directly to unrotated Q/K.

        # Here, we treat Qh/Kh as the outputs of the projection + RoPE.



        # 7. Backprop through split_heads and W_q/W_k/W_v projections

        dQ = np.transpose(dQh, (0, 2, 1, 3)).reshape(b, s, d) # (B, S, D)

        dK = np.transpose(dKh, (0, 2, 1, 3)).reshape(b, s, d) # (B, S, D)

        dV = np.transpose(dVh, (0, 2, 1, 3)).reshape(b, s, d) # (B, S, D)



        # Accumulate gradients for weight matrices

        x_flat = x.reshape(-1, d)

        self.grad_W_q = x_flat.T @ dQ.reshape(-1, d)

        self.grad_W_k = x_flat.T @ dK.reshape(-1, d)

        self.grad_W_v = x_flat.T @ dV.reshape(-1, d)



        # 8. Backprop to input x (sum of gradients from Q, K, V paths)

        dx_q = dQ @ self.W_q.T

        dx_k = dK @ self.W_k.T

        dx_v = dV @ self.W_v.T

        dx = dx_q + dx_k + dx_v

        return dx



class DecoderBlock:

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, 

                 layer_scale: bool = False, layer_scale_init: float = 1e-4, 

                 use_rotary: bool = False, rms_norm: bool = False):

        self.d_model = d_model

        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout, use_rotary)

        self.ff = FeedForward(d_model, d_ff, dropout)

        self.ln1 = LayerNorm(d_model, rms_norm=rms_norm)

        self.ln2 = LayerNorm(d_model, rms_norm=rms_norm)

        self.layer_scale = layer_scale

        

        if layer_scale:

            # LayerScale parameters (initialized small)

            self.gamma1 = np.ones((1, 1, d_model)) * layer_scale_init

            self.gamma2 = np.ones((1, 1, d_model)) * layer_scale_init

            self.grad_gamma1 = np.zeros_like(self.gamma1)

            self.grad_gamma2 = np.zeros_like(self.gamma2)

            # Cache for backward pass

            self.attn_out_pre_gamma = None 

            self.ff_out_pre_gamma = None

        else:

            self.gamma1 = self.gamma2 = None

            self.grad_gamma1 = self.grad_gamma2 = None

            self.attn_out_pre_gamma = self.ff_out_pre_gamma = None



    def forward(self, x: np.ndarray, pos_freqs: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:

        # --- Self-Attention Block ---

        attn_out = self.mha.forward(x, pos_freqs, training)

        

        if self.layer_scale:

            self.attn_out_pre_gamma = attn_out # Cache before scaling

            attn_out = attn_out * self.gamma1

            

        x = x + attn_out # Residual connection 1

        x = self.ln1.forward(x)

        

        # --- Feed-Forward Block ---

        ff_out = self.ff.forward(x, training)

        

        if self.layer_scale:

            self.ff_out_pre_gamma = ff_out # Cache before scaling

            ff_out = ff_out * self.gamma2

            

        x = x + ff_out # Residual connection 2

        x = self.ln2.forward(x)

        

        return x



    def backward(self, grad: np.ndarray) -> np.ndarray:

        # 1. Backprop through LayerNorm 2

        d_res2 = self.ln2.backward(grad)

        

        # 2. Backprop through FF block (x + ff_out)

        d_ff_pre_gamma = d_res2.copy() # Gradient flowing to FF output

        if self.layer_scale:

            # Calculate gradient for gamma2: dL/d(gamma2) = sum(d_ff_pre_gamma * ff_out_pre_gamma)

            self.grad_gamma2 = np.sum(d_ff_pre_gamma * self.ff_out_pre_gamma, axis=(0, 1), keepdims=True)

            d_ff_pre_gamma = d_ff_pre_gamma * self.gamma2 # Scale back gradient by gamma2

            

        d_ff = self.ff.backward(d_ff_pre_gamma)

        d_res1 = d_ff + d_res2 # Gradient flowing into LayerNorm1 (sum of FF grad and residual grad)

        

        # 3. Backprop through LayerNorm 1

        d_ln1 = self.ln1.backward(d_res1)

        

        # 4. Backprop through MHA block (x + attn_out)

        d_attn_pre_gamma = d_ln1.copy() # Gradient flowing to MHA output

        if self.layer_scale:

            # Calculate gradient for gamma1: dL/d(gamma1) = sum(d_attn_pre_gamma * attn_out_pre_gamma)

            self.grad_gamma1 = np.sum(d_attn_pre_gamma * self.attn_out_pre_gamma, axis=(0, 1), keepdims=True)

            d_attn_pre_gamma = d_attn_pre_gamma * self.gamma1 # Scale back gradient by gamma1

            

        d_mha = self.mha.backward(d_attn_pre_gamma)

        

        # 5. Final dx is the sum of gradients from MHA and the residual connection

        dx = d_mha + d_ln1

        return dx



class GPT:

    """A minimal GPT decoder-only model implementation using only NumPy."""

    def __init__(self, vocab_size: int, max_len: int = 512, d_model: int = 768, num_heads: int = 12, 

                 d_ff: int = 3072, num_layers: int = 12, dropout: float = 0.1, 

                 use_rotary: bool = False, rms_norm: bool = False, layer_scale: bool = False,

                 dtype=DEFAULT_DTYPE):

        self.vocab_size = vocab_size

        self.max_len = max_len

        self.d_model = d_model

        self.dtype = dtype

        

        self.embed = Embedding(vocab_size, d_model, dtype)

        self.pos_embed = PositionalEmbedding(max_len, d_model, use_rotary, dtype)

        

        self.layers = [

            DecoderBlock(d_model, num_heads, d_ff, dropout, layer_scale, 

                         use_rotary=use_rotary, rms_norm=rms_norm)

            for _ in range(num_layers)

        ]

        

        self.ln_f = LayerNorm(d_model, rms_norm=rms_norm, dtype=dtype)

        self.dropout = dropout

        

        # Output projection layer (Tied weights often used, but here we keep W_out separate)

        self.W_out = np.random.normal(0, 1.0 / np.sqrt(d_model), (d_model, vocab_size)).astype(dtype)

        self.grad_W_out = np.zeros_like(self.W_out)

        

        # Optimization State

        self.opt_states: Dict[str, Dict[str, np.ndarray]] = {}

        self.lr = 0.0

        self.beta1 = 0.0

        self.beta2 = 0.0

        self.eps = 0.0

        self.opt_step = 0

        self.training = True

        

    def parameters(self) -> List[Tuple[str, np.ndarray]]:

        """Returns a list of (name, parameter_array) tuples for all learnable parameters."""

        params = []

        params.append(('embed.W', self.embed.W))

        if not self.pos_embed.use_rotary:

            params.append(('pos.W', self.pos_embed.W))

        

        for i, layer in enumerate(self.layers):

            # MHA Weights

            params.append((f'layer{i}.mha.W_q', layer.mha.W_q))

            params.append((f'layer{i}.mha.W_k', layer.mha.W_k))

            params.append((f'layer{i}.mha.W_v', layer.mha.W_v))

            params.append((f'layer{i}.mha.W_o', layer.mha.W_o))

            

            # LayerNorm 1 Weights

            if not layer.ln1.rms_norm:

                params.append((f'layer{i}.ln1.gamma', layer.ln1.gamma))

                params.append((f'layer{i}.ln1.beta', layer.ln1.beta))

            else:

                params.append((f'layer{i}.ln1.weight', layer.ln1.weight))

                

            # FF Weights

            params.append((f'layer{i}.ff.W1', layer.ff.W1))

            params.append((f'layer{i}.ff.b1', layer.ff.b1))

            params.append((f'layer{i}.ff.W2', layer.ff.W2))

            params.append((f'layer{i}.ff.b2', layer.ff.b2))

            

            # LayerNorm 2 Weights

            if not layer.ln2.rms_norm:

                params.append((f'layer{i}.ln2.gamma', layer.ln2.gamma))

                params.append((f'layer{i}.ln2.beta', layer.ln2.beta))

            else:

                params.append((f'layer{i}.ln2.weight', layer.ln2.weight))

                

            # LayerScale Weights

            if layer.layer_scale:

                params.append((f'layer{i}.gamma1', layer.gamma1))

                params.append((f'layer{i}.gamma2', layer.gamma2))



        # Final LayerNorm Weights

        if not self.ln_f.rms_norm:

            params.append(('ln_f.gamma', self.ln_f.gamma))

            params.append(('ln_f.beta', self.ln_f.beta))

        else:

            params.append(('ln_f.weight', self.ln_f.weight))

            

        params.append(('W_out', self.W_out))

        return params



    def get_grad_by_name(self, name: str) -> np.ndarray:

        """Retrieves the gradient array corresponding to a parameter name."""

        if name == 'embed.W': return self.embed.grad_W

        if name == 'pos.W': 

            if self.pos_embed.use_rotary: raise ValueError("RoPE enabled, pos.W has no gradient.")

            return self.pos_embed.grad_W

        if name == 'W_out': return self.grad_W_out

        

        # Final LayerNorm

        if name == 'ln_f.gamma': return self.ln_f.grad_gamma

        if name == 'ln_f.beta': return self.ln_f.grad_beta

        if name == 'ln_f.weight': return self.ln_f.grad_weight



        # Layer parameters

        parts = name.split('.')

        layer_idx = int(parts[0].replace('layer', ''))

        layer = self.layers[layer_idx]

        param_part = '.'.join(parts[1:])

        

        if param_part == 'mha.W_q': return layer.mha.grad_W_q

        if param_part == 'mha.W_k': return layer.mha.grad_W_k

        if param_part == 'mha.W_v': return layer.mha.grad_W_v

        if param_part == 'mha.W_o': return layer.mha.grad_W_o

        

        if param_part == 'ff.W1': return layer.ff.grad_W1

        if param_part == 'ff.b1': return layer.ff.grad_b1

        if param_part == 'ff.W2': return layer.ff.grad_W2

        if param_part == 'ff.b2': return layer.ff.grad_b2



        if layer.layer_scale:

            if param_part == 'gamma1': return layer.grad_gamma1

            if param_part == 'gamma2': return layer.grad_gamma2



        if layer.ln1.rms_norm:

            if param_part == 'ln1.weight': return layer.ln1.grad_weight

        else:

            if param_part == 'ln1.gamma': return layer.ln1.grad_gamma

            if param_part == 'ln1.beta': return layer.ln1.grad_beta

            

        if layer.ln2.rms_norm:

            if param_part == 'ln2.weight': return layer.ln2.grad_weight

        else:

            if param_part == 'ln2.gamma': return layer.ln2.grad_gamma

            if param_part == 'ln2.beta': return layer.ln2.grad_beta

            

        raise ValueError(f"Unknown parameter name: {name}")



    def zero_grads(self):

        """Resets all stored gradients to zero."""

        self.embed.grad_W.fill(0.0)

        if not self.pos_embed.use_rotary:

            self.pos_embed.grad_W.fill(0.0)

        self.grad_W_out.fill(0.0)



        # Clear gradients in LayerNorm components

        if not self.ln_f.rms_norm:

            self.ln_f.grad_gamma.fill(0.0)

            self.ln_f.grad_beta.fill(0.0)

        else:

            self.ln_f.grad_weight.fill(0.0)



        # Clear gradients in all layers

        for layer in self.layers:

            # MHA

            layer.mha.grad_W_q.fill(0.0)

            layer.mha.grad_W_k.fill(0.0)

            layer.mha.grad_W_v.fill(0.0)

            layer.mha.grad_W_o.fill(0.0)

            # FF

            layer.ff.grad_W1.fill(0.0)

            layer.ff.grad_b1.fill(0.0)

            layer.ff.grad_W2.fill(0.0)

            layer.ff.grad_b2.fill(0.0)

            # LayerNorms

            if not layer.ln1.rms_norm:

                layer.ln1.grad_gamma.fill(0.0)

                layer.ln1.grad_beta.fill(0.0)

            else:

                layer.ln1.grad_weight.fill(0.0)

            if not layer.ln2.rms_norm:

                layer.ln2.grad_gamma.fill(0.0)

                layer.ln2.grad_beta.fill(0.0)

            else:

                layer.ln2.grad_weight.fill(0.0)

            # LayerScale

            if layer.layer_scale:

                layer.grad_gamma1.fill(0.0)

                layer.grad_gamma2.fill(0.0)



    def forward(self, idx: np.ndarray, training: bool = True) -> np.ndarray:

        self.training = training

        b, s = idx.shape

        x = self.embed.forward(idx)

        

        pos_embed = self.pos_embed.forward(s)

        if pos_embed is not None:

            x = x + pos_embed

        

        pos_freqs = self.pos_embed.rotary_freqs if self.pos_embed.use_rotary else None

        

        for layer in self.layers:

            x = layer.forward(x, pos_freqs, training)

            

        x = self.ln_f.forward(x)

        

        # Final Dropout

        if training and self.dropout > 0:

            dropout_mask = (np.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)

            x = x * dropout_mask

            

        # Output Projection

        logits = x.reshape(-1, self.d_model) @ self.W_out

        logits = logits.reshape(b, s, -1)

        

        self._cache = {'x': x, 'idx': idx, 'dropout_mask': dropout_mask if training else None}

        return logits



    def loss_and_backward(self, idx_in: np.ndarray, idx_target: np.ndarray, 

                          grad_clip: float = 1.0) -> float:

        """Calculates cross-entropy loss and performs a full backward pass."""

        b, s = idx_in.shape

        logits = self.forward(idx_in, training=True)

        vocab = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab)

        targets_flat = idx_target.reshape(-1)

        

        # Compute Cross-Entropy Loss

        probs = softmax(logits_flat, axis=1)

        log_probs = np.log(np.clip(probs, 1e-12, 1.0))

        loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])

        

        # Start Backward Pass: Gradient of Loss w.r.t logits (dLoss/dLogits)

        grad_logits = probs.copy()

        grad_logits[np.arange(grad_logits.shape[0]), targets_flat] -= 1

        grad_logits = grad_logits.reshape(b, s, vocab) / (b * s) # Normalize by batch * sequence length



        # 1. Backprop through W_out

        x = self._cache['x']

        self.grad_W_out = x.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, vocab)

        dx = grad_logits.reshape(-1, vocab) @ self.W_out.T

        dx = dx.reshape(b, s, self.d_model)

        

        # 2. Backprop through final Dropout

        if self._cache['dropout_mask'] is not None:

            dx = dx * self._cache['dropout_mask']



        # 3. Backprop through Final LayerNorm

        d_ln = self.ln_f.backward(dx)

        grad = d_ln

        

        # 4. Backprop through Decoder Blocks (in reverse)

        for layer in reversed(self.layers):

            grad = layer.backward(grad)

            

        # 5. Backprop through Embedding (The input to the first block)

        idx = self._cache['idx']

        self.embed.backward(idx, grad)

        

        # 6. Backprop through Positional Embedding (if not using RoPE)

        if not self.pos_embed.use_rotary:

            self.pos_embed.backward(s, grad)



        # 7. Gradient Clipping (Global Norm)

        if grad_clip > 0:

            total_norm_sq = 0.0

            for name, _ in self.parameters():

                grad_array = self.get_grad_by_name(name)

                total_norm_sq += np.sum(grad_array ** 2)

            total_norm = np.sqrt(total_norm_sq)

            

            clip_coef = min(grad_clip / (total_norm + EPS), 1.0)

            if clip_coef < 1.0:

                for name, _ in self.parameters():

                    grad_array = self.get_grad_by_name(name)

                    grad_array *= clip_coef



        return loss



    def init_optimizer(self, lr: float = 6e-4, betas=(0.9, 0.95), eps=1e-8, 

                       weight_decay: float = 0.1, warmup_steps: int = 2000):

        """Initializes AdamW optimizer states."""

        self.lr = lr

        self.beta1 = betas[0]

        self.beta2 = betas[1]

        self.eps = eps

        self.weight_decay = weight_decay

        self.warmup_steps = warmup_steps

        self.opt_step = 0

        self.opt_states = {}

        

        # Initialize moments (m and v) for every parameter

        for name, param in self.parameters():

            self.opt_states[name] = {

                'm': np.zeros_like(param),

                'v': np.zeros_like(param)

            }



    def step_optimizer(self, current_step: Optional[int] = None):

        """Performs one step of the AdamW optimization algorithm."""

        if current_step is not None:

            self.opt_step = current_step

        self.opt_step += 1



        # Calculate learning rate with Warmup and Inverse Square Root decay

        if self.warmup_steps > 0:

            lr = self.lr * min(self.opt_step ** -0.5, self.opt_step * self.warmup_steps ** -1.5)

        else:

            lr = self.lr

        

        lr_t = lr

        

        for name, param in self.parameters():

            grad = self.get_grad_by_name(name)

            state = self.opt_states[name]

            

            # 1. Apply Weight Decay (Exclude biases and LayerNorm/LayerScale/Positional embeddings)

            if 'W' in name and 'embed.W' not in name and 'pos.W' not in name and self.weight_decay > 0:

                 grad = grad + self.weight_decay * param

            

            # 2. Update biased first and second moment estimates (Adam)

            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad

            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)



            # 3. Compute bias-corrected moments (using current step t)

            m_hat = state['m'] / (1 - self.beta1 ** self.opt_step)

            v_hat = state['v'] / (1 - self.beta2 ** self.opt_step)

            

            # 4. Update parameters

            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)



    def save(self, path: str, include_optimizer: bool = False):

        """Saves model configuration and parameters to a file using pickle."""

        data: Dict[str, Any] = {

            'config': {

                'vocab_size': self.vocab_size,

                'max_len': self.max_len,

                'd_model': self.d_model,

                'num_heads': self.layers[0].mha.num_heads,

                'd_ff': self.layers[0].ff.d_ff,

                'num_layers': len(self.layers),

                'dropout': self.dropout,

                'use_rotary': self.pos_embed.use_rotary,

                'rms_norm': self.ln_f.rms_norm,

                'layer_scale': any(layer.layer_scale for layer in self.layers)

            },

            # Save core embeddings and output layer

            'embed.W': self.embed.W,

            'pos.W': self.pos_embed.W if not self.pos_embed.use_rotary else None,

            'W_out': self.W_out,

            # Save final LayerNorm

            'ln_f': {

                'gamma': self.ln_f.gamma, 

                'beta': self.ln_f.beta, 

                'weight': self.ln_f.weight

            },

            'layers': []

        }

        

        # Save individual decoder block parameters

        for layer in self.layers:

            layer_data = {

                'mha.W_q': layer.mha.W_q,

                'mha.W_k': layer.mha.W_k,

                'mha.W_v': layer.mha.W_v,

                'mha.W_o': layer.mha.W_o,

                'ff.W1': layer.ff.W1,

                'ff.b1': layer.ff.b1,

                'ff.W2': layer.ff.W2,

                'ff.b2': layer.ff.b2,

                'ln1': {'gamma': layer.ln1.gamma, 'beta': layer.ln1.beta, 'weight': layer.ln1.weight},

                'ln2': {'gamma': layer.ln2.gamma, 'beta': layer.ln2.beta, 'weight': layer.ln2.weight},

                'layer_scale': {}

            }

            if layer.layer_scale:

                layer_data['layer_scale']['gamma1'] = layer.gamma1

                layer_data['layer_scale']['gamma2'] = layer.gamma2

            data['layers'].append(layer_data)

        

        # Save optimizer state if requested

        if include_optimizer and self.opt_states:

            data['optimizer'] = {

                'lr': self.lr, 'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps,

                'weight_decay': self.weight_decay, 'warmup_steps': self.warmup_steps,

                'opt_step': self.opt_step,

                'states': self.opt_states

            }



        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

        with open(path, 'wb') as f:

            pickle.dump(data, f)

            

    def load(self, path: str, strict: bool = True):

        """Loads model parameters from a saved file."""

        with open(path, 'rb') as f:

            data = pickle.load(f)



        # Check config (simplified check)

        loaded_config = data['config']

        current_config = {

            'vocab_size': self.vocab_size, 'max_len': self.max_len, 'd_model': self.d_model,

            'num_layers': len(self.layers)

        }

        if strict:

            for key, val in current_config.items():

                if loaded_config.get(key) != val:

                    warnings.warn(f"Configuration mismatch on key '{key}': Current={val}, Loaded={loaded_config.get(key)}", RuntimeWarning)



        # Load core embeddings and output layer

        self.embed.W = data['embed.W']

        if not self.pos_embed.use_rotary and data['pos.W'] is not None:

            self.pos_embed.W = data['pos.W']

        self.W_out = data['W_out']

        

        # Load final LayerNorm

        ln_f_data = data['ln_f']

        if not self.ln_f.rms_norm and ln_f_data.get('gamma') is not None:

            self.ln_f.gamma = ln_f_data['gamma']

            self.ln_f.beta = ln_f_data['beta']

        elif self.ln_f.rms_norm and ln_f_data.get('weight') is not None:

            self.ln_f.weight = ln_f_data['weight']



        # Load individual decoder block parameters

        for i, layer_data in enumerate(data['layers']):

            layer = self.layers[i]

            

            # MHA

            layer.mha.W_q = layer_data['mha.W_q']

            layer.mha.W_k = layer_data['mha.W_k']

            layer.mha.W_v = layer_data['mha.W_v']

            layer.mha.W_o = layer_data['mha.W_o']

            

            # FF

            layer.ff.W1 = layer_data['ff.W1']

            layer.ff.b1 = layer_data['ff.b1']

            layer.ff.W2 = layer_data['ff.W2']

            layer.ff.b2 = layer_data['ff.b2']



            # LayerNorm 1

            ln1_data = layer_data['ln1']

            if not layer.ln1.rms_norm and ln1_data.get('gamma') is not None:

                layer.ln1.gamma = ln1_data['gamma']

                layer.ln1.beta = ln1_data['beta']

            elif layer.ln1.rms_norm and ln1_data.get('weight') is not None:

                layer.ln1.weight = ln1_data['weight']

                

            # LayerNorm 2

            ln2_data = layer_data['ln2']

            if not layer.ln2.rms_norm and ln2_data.get('gamma') is not None:

                layer.ln2.gamma = ln2_data['gamma']

                layer.ln2.beta = ln2_data['beta']

            elif layer.ln2.rms_norm and ln2_data.get('weight') is not None:

                layer.ln2.weight = ln2_data['weight']

                

            # LayerScale

            if layer.layer_scale and 'layer_scale' in layer_data:

                layer.gamma1 = layer_data['layer_scale']['gamma1']

                layer.gamma2 = layer_data['layer_scale']['gamma2']



        # Load optimizer state

        if 'optimizer' in data:

            opt_data = data['optimizer']

            self.init_optimizer(

                lr=opt_data['lr'], betas=(opt_data['beta1'], opt_data['beta2']), eps=opt_data['eps'],

                weight_decay=opt_data['weight_decay'], warmup_steps=opt_data['warmup_steps']

            )

            self.opt_step = opt_data['opt_step']

            self.opt_states = opt_data['states']



# --- Example Usage (Conceptual) ---

if __name__ == '__main__':

    # 1. Prepare Dummy Data and Tokenizer

    corpus = [

        "سلام، دنیای هوش مصنوعی. این یک تست است.",

        "آموزش شبکه عصبی با NumPy یک چالش جالب است."

    ]

    tokenizer = BPETokenizer()

    tokenizer.build_from_text(corpus, vocab_size=100)

    

    VOCAB_SIZE = len(tokenizer.vocab)

    MAX_LEN = 10

    BATCH_SIZE = 2

    D_MODEL = 64

    NUM_LAYERS = 2

    NUM_HEADS = 4

    D_FF = 256

    

    # Encode dummy data

    inputs = np.array([tokenizer.encode(t, max_len=MAX_LEN) for t in corpus])

    

    # Create target array (shifted input for language modeling)

    idx_in = inputs[:, :-1]

    idx_target = inputs[:, 1:]



    print(f"Vocab Size: {VOCAB_SIZE}")

    print(f"Input Shape (B, S): {idx_in.shape}")

    print("--- Initializing GPT Model ---")

    

    # 2. Initialize Model

    model = GPT(

        vocab_size=VOCAB_SIZE, 

        max_len=MAX_LEN, 

        d_model=D_MODEL, 

        num_heads=NUM_HEADS, 

        d_ff=D_FF, 

        num_layers=NUM_LAYERS,

        use_rotary=True, 

        rms_norm=True, 

        layer_scale=True

    )

    

    # 3. Initialize Optimizer

    model.init_optimizer(lr=1e-3)



    # 4. Training Loop (Single Step Example)

    print("--- Performing Single Training Step ---")

    

    model.zero_grads()

    start_time = time.time()

    loss = model.loss_and_backward(idx_in, idx_target, grad_clip=1.0)

    end_loss_time = time.time()

    model.step_optimizer()

    end_step_time = time.time()



    print(f"Loss: {loss:.4f}")

    print(f"Forward/Backward Time: {end_loss_time - start_time:.4f}s")

    print(f"Optimizer Step Time: {end_step_time - end_loss_time:.4f}s")



    # 5. Inference Example

    print("--- Performing Inference ---")

    dummy_input = inputs[:1, :-1]

    logits_inference = model.forward(dummy_input, training=False)

    predicted_id = np.argmax(logits_inference[0, -1, :])

    predicted_token = tokenizer.decode([predicted_id])

    

    print(f"Input text (decoded): {tokenizer.decode(dummy_input[0])}")

    print(f"Predicted next token (decoded): {predicted_token}")



    # 6. Save and Load Test

    save_path = './numpy_gpt_model.pkl'

    model.save(save_path, include_optimizer=True)

    print(f"Model saved to {save_path}")

    

    new_model = GPT(VOCAB_SIZE, MAX_LEN, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)

    new_model.load(save_path)

    print("Model loaded successfully.")

    # You would need to re-initialize or load the optimizer state separately if needed.

    os.remove(save_path)
