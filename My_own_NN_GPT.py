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
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
DEFAULT_DTYPE = np.float32
EPS = 1e-6

def softmax(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + eps)

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + math.erf(x / np.sqrt(2.0)))

def gelu_grad(x: np.ndarray) -> np.ndarray:
    tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    sech2 = 1.0 - tanh_term**2
    return 0.5 * (1.0 + tanh_term) + 0.5 * x * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return weight * (x / rms)

class BPETokenizer:
    def __init__(self):
        self.vocab: List[str] = []
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.cache: Dict[str, List[str]] = {}
        self.special_tokens: List[str] = ['<pad>', '<unk>', '<bos>', '<eos>']

    @staticmethod
    def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        return set(zip(word, word[1:]))

    @staticmethod
    def bytes_to_unicode() -> Dict[int, str]:
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
        byte_encoder = self.bytes_to_unicode()
        text_bytes = text.encode("utf-8")
        return "".join([byte_encoder[b] for b in text_bytes])

    def build_from_text(self, texts: List[str], vocab_size: int = 500, min_freq: int = 2):
        preprocessed = [self.preprocess(text) for text in texts]
        char_freq = {}
        for text in preprocessed:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        self.vocab = self.special_tokens + sorted(char_freq.keys(), key=lambda x: -char_freq[x])
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}
        if len(self.vocab) < vocab_size:
            words = []
            for text in preprocessed:
                words.extend([' '.join(text)])
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            num_merges = vocab_size - len(self.vocab)
            for i in range(num_merges):
                pairs = {}
                for word, freq in word_freq.items():
                    chars = word.split()
                    for j in range(len(chars) - 1):
                        pair = (chars[j], chars[j+1])
                        pairs[pair] = pairs.get(pair, 0) + freq
                if not pairs:
                    break
                best_pair = max(pairs, key=pairs.get)
                new_token = ''.join(best_pair)
                if new_token not in self.w2i:
                    self.vocab.append(new_token)
                    self.w2i[new_token] = len(self.vocab) - 1
                    self.i2w[len(self.vocab) - 1] = new_token
                    self.merges.append(best_pair)
                new_word_freq = {}
                for word, freq in word_freq.items():
                    new_word = word.replace(' '.join(best_pair), new_token)
                    new_word_freq[new_word] = freq
                word_freq = new_word_freq

    def encode(self, text: str, max_len: int = None, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        text = self.preprocess(text)
        if add_bos:
            text = self.special_tokens[2] + text
        if add_eos:
            text = text + self.special_tokens[3]
        if text in self.cache:
            tokens = self.cache[text]
        else:
            tokens = list(text)
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
            self.cache[text] = tokens
        ids = [self.w2i.get(t, self.w2i['<unk>']) for t in tokens]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        if max_len is not None and len(ids) < max_len:
            ids = ids + [self.w2i['<pad>']] * (max_len - len(ids))
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: Union[np.ndarray, List[int]]) -> str:
        tokens = [self.i2w.get(int(i), '<unk>') for i in ids]
        text = ''.join(tokens)
        for token in self.special_tokens:
            text = text.replace(token, '')
        byte_decoder = {v: k for k, v in self.bytes_to_unicode().items()}
        text_bytes = bytearray([byte_decoder[c] for c in text])
        return text_bytes.decode('utf-8', errors='replace')

class Embedding:
    def __init__(self, vocab_size: int, d_model: int, dtype=DEFAULT_DTYPE):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        scale = 1.0 / np.sqrt(d_model)
        self.W = np.random.normal(0, scale, (vocab_size, d_model)).astype(dtype)
        self.grad_W = np.zeros_like(self.W)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        return self.W[idx]

    def backward(self, idx: np.ndarray, grad: np.ndarray):
        np.add.at(self.grad_W, idx, grad)

class PositionalEmbedding:
    def __init__(self, max_len: int, d_model: int, use_rotary: bool = False, dtype=DEFAULT_DTYPE):
        self.max_len = max_len
        self.d_model = d_model
        self.use_rotary = use_rotary
        self.dtype = dtype
        if not use_rotary:
            self.W = np.zeros((max_len, d_model), dtype=dtype)
            for pos in range(max_len):
                for i in range(0, d_model, 2):
                    self.W[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                    if i + 1 < d_model:
                        self.W[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
            self.grad_W = np.zeros_like(self.W)
        else:
            self.rotary_freqs = self._create_rotary_frequencies()

    def _create_rotary_frequencies(self) -> np.ndarray:
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.d_model, 2, dtype=self.dtype) / self.d_model))
        return inv_freq

    def apply_rotary_pos_emb(self, x: np.ndarray, seq_dim: int = -2) -> np.ndarray:
        seq_len = x.shape[seq_dim]
        t = np.arange(seq_len, dtype=self.dtype)
        freqs = np.outer(t, self.rotary_freqs)
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rotated1 = x1 * cos - x2 * sin
        x_rotated2 = x1 * sin + x2 * cos
        x_rotated = np.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated1
        x_rotated[..., 1::2] = x_rotated2
        return x_rotated

    def forward(self, seq_len: int) -> np.ndarray:
        if not self.use_rotary:
            return self.W[:seq_len][np.newaxis, :, :]
        return None

    def backward(self, seq_len: int, grad: np.ndarray):
        if not self.use_rotary:
            np.add.at(self.grad_W, np.arange(seq_len), np.sum(grad, axis=0))

class LayerNorm:
    def __init__(self, d_model: int, eps: float = EPS, rms_norm: bool = False, dtype=DEFAULT_DTYPE):
        self.d_model = d_model
        self.eps = eps
        self.rms_norm = rms_norm
        self.dtype = dtype
        if not rms_norm:
            self.gamma = np.ones((1, 1, d_model), dtype=dtype)
            self.beta = np.zeros((1, 1, d_model), dtype=dtype)
            self.grad_gamma = np.zeros_like(self.gamma)
            self.grad_beta = np.zeros_like(self.beta)
        else:
            self.weight = np.ones((1, 1, d_model), dtype=dtype)
            self.grad_weight = np.zeros_like(self.weight)
        self.x = None
        self.mean = None
        self.var = None
        self.x_norm = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.rms_norm:
            rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
            self.x_norm = x / rms
            return self.weight * self.x_norm
        else:
            self.mean = np.mean(x, axis=-1, keepdims=True)
            self.var = np.var(x, axis=-1, keepdims=True)
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            return self.gamma * self.x_norm + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.rms_norm:
            grad_x_norm = grad * self.weight
            x_norm2 = self.x_norm ** 2
            d_rms = -np.sum(grad_x_norm * self.x_norm, axis=-1, keepdims=True) / np.sqrt(np.mean(x_norm2, axis=-1, keepdims=True) + self.eps)
            d_x = (grad_x_norm - self.x_norm * d_rms) / self.x_norm.shape[-1]
            self.grad_weight = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)
            return d_x
        else:
            b, s, d = grad.shape
            self.grad_gamma = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)
            self.grad_beta = np.sum(grad, axis=(0, 1), keepdims=True)
            dx_norm = grad * self.gamma
            var_eps = self.var + self.eps
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
        self.W1 = np.random.normal(0, scale_in, (d_model, d_ff)).astype(dtype)
        self.b1 = np.zeros((1, 1, d_ff), dtype=dtype)
        self.W2 = np.random.normal(0, scale_out, (d_ff, d_model)).astype(dtype)
        self.b2 = np.zeros((1, 1, d_model), dtype=dtype)
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
        self.x = None
        self.hidden = None
        self.hidden_act = None
        self.dropout_mask1 = None
        self.dropout_mask2 = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.x = x
        b, s, d = x.shape
        self.hidden = x @ self.W1 + self.b1
        self.hidden_act = gelu(self.hidden)
        if training and self.dropout > 0:
            self.dropout_mask1 = (np.random.rand(*self.hidden_act.shape) > self.dropout)
            self.hidden_act = self.hidden_act * self.dropout_mask1 / (1 - self.dropout)
        else:
            self.dropout_mask1 = None
        out = self.hidden_act @ self.W2 + self.b2
        if training and self.dropout > 0:
            self.dropout_mask2 = (np.random.rand(*out.shape) > self.dropout)
            out = out * self.dropout_mask2 / (1 - self.dropout)
        else:
            self.dropout_mask2 = None
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, s, d = grad.shape
        if self.dropout_mask2 is not None:
            grad = grad * self.dropout_mask2
        self.grad_W2 = (self.hidden_act.reshape(-1, self.d_ff).T @ grad.reshape(-1, d)).reshape(self.d_ff, d)
        self.grad_b2 = np.sum(grad, axis=(0, 1), keepdims=True)
        dhidden_act = grad @ self.W2.T
        if self.dropout_mask1 is not None:
            dhidden_act = dhidden_act * self.dropout_mask1
        dhidden = dhidden_act * gelu_grad(self.hidden)
        self.grad_W1 = (self.x.reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_ff)).reshape(self.d_model, self.d_ff)
        self.grad_b1 = np.sum(dhidden, axis=(0, 1), keepdims=True)
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
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)
        self.W_k = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)
        self.W_v = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)
        self.W_o = np.random.normal(0, scale, (d_model, d_model)).astype(dtype)
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.cache = {}
        self.dropout_mask = None

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        b, s, d = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        x = np.transpose(x, (0, 2, 1, 3))
        b, s, h, hd = x.shape
        return x.reshape(b, s, h * hd)

    def causal_mask(self, seq_len: int) -> np.ndarray:
        return np.tril(np.ones((seq_len, seq_len), dtype=bool))

    def apply_rotary_embeddings(self, q: np.ndarray, k: np.ndarray, seq_dim: int = -2) -> Tuple[np.ndarray, np.ndarray]:
        q_rotated = PositionalEmbedding.apply_rotary_pos_emb(q, seq_dim=seq_dim)
        k_rotated = PositionalEmbedding.apply_rotary_pos_emb(k, seq_dim=seq_dim)
        return q_rotated, k_rotated

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        b, s, d = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Qh = self.split_heads(Q)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)
        if self.use_rotary:
            Qh, Kh = self.apply_rotary_embeddings(Qh, Kh)
        dk = self.head_dim
        scores = Qh @ np.swapaxes(Kh, -1, -2) / np.sqrt(dk)
        mask = self.causal_mask(s)[np.newaxis, np.newaxis, :, :]
        scores = np.where(mask, scores, -np.inf)
        attn = softmax(scores, axis=-1)
        if training and self.dropout > 0:
            self.dropout_mask = (np.random.rand(*attn.shape) > self.dropout)
            attn = attn * self.dropout_mask / (1 - self.dropout)
        else:
            self.dropout_mask = None
        attn_out = attn @ Vh
        out = self.combine_heads(attn_out) @ self.W_o
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'Qh': Qh, 'Kh': Kh, 'Vh': Vh,
            'scores': scores, 'attn': attn, 'attn_out': attn_out,
            'mask': mask
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        Qh = self.cache['Qh']
        Kh = self.cache['Kh']
        Vh = self.cache['Vh']
        attn = self.cache['attn']
        attn_out = self.cache['attn_out']
        mask = self.cache['mask']
        b, s, d = grad_out.shape
        dk = self.head_dim
        if self.dropout_mask is not None:
            attn = attn * self.dropout_mask
        out_concat = self.combine_heads(attn_out)
        self.grad_W_o = out_concat.reshape(-1, d).T @ grad_out.reshape(-1, d)
        d_out_concat = grad_out @ self.W_o.T
        d_attn_out = d_out_concat.reshape(b, s, self.num_heads, self.head_dim)
        d_attn_out = np.transpose(d_attn_out, (0, 2, 1, 3))
        dVh = np.matmul(np.swapaxes(attn, -1, -2), d_attn_out)
        dattn = np.matmul(d_attn_out, np.swapaxes(Vh, -1, -2))
        sft = attn
        sum_d = np.sum(dattn * sft, axis=-1, keepdims=True)
        dscores = sft * (dattn - sum_d)
        dscores = np.where(mask, dscores, 0.0)
        dQh = np.matmul(dscores, Kh) / np.sqrt(dk)
        dKh = np.matmul(np.swapaxes(dscores, -1, -2), Qh) / np.sqrt(dk)
        dQ = np.transpose(dQh, (0, 2, 1, 3)).reshape(b, s, d)
        dK = np.transpose(dKh, (0, 2, 1, 3)).reshape(b, s, d)
        dV = np.transpose(dVh, (0, 2, 1, 3)).reshape(b, s, d)
        self.grad_W_q = x.reshape(-1, d).T @ dQ.reshape(-1, d)
        self.grad_W_k = x.reshape(-1, d).T @ dK.reshape(-1, d)
        self.grad_W_v = x.reshape(-1, d).T @ dV.reshape(-1, d)
        dx_q = dQ @ self.W_q.T
        dx_k = dK @ self.W_k.T
        dx_v = dV @ self.W_v.T
        dx = dx_q + dx_k + dx_v
        return dx

class DecoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, 
                 layer_scale: bool = False, layer_scale_init: float = 1e-4, use_rotary: bool = False):
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout, use_rotary)
        self.ln1 = LayerNorm(d_model, rms_norm=False)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = LayerNorm(d_model, rms_norm=False)
        self.dropout = dropout
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        if layer_scale:
            self.gamma1 = np.ones((1, 1, d_model)) * layer_scale_init
            self.gamma2 = np.ones((1, 1, d_model)) * layer_scale_init

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        attn_out = self.mha.forward(x, training)
        if self.layer_scale:
            attn_out = attn_out * self.gamma1
        x = x + attn_out
        x = self.ln1.forward(x)
        ff_out = self.ff.forward(x, training)
        if self.layer_scale:
            ff_out = ff_out * self.gamma2
        x = x + ff_out
        x = self.ln2.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_ln2 = self.ln2.backward(grad)
        d_ff = self.ff.backward(d_ln2)
        if self.layer_scale:
            d_ff = d_ff * self.gamma2
        d_res = d_ln2 + d_ff
        d_ln1 = self.ln1.backward(d_res)
        d_mha = self.mha.backward(d_ln1)
        if self.layer_scale:
            d_mha = d_mha * self.gamma1
        dx = d_mha + d_ln1
        return dx

class GPT:
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
            DecoderBlock(d_model, num_heads, d_ff, dropout, layer_scale, use_rotary=use_rotary)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(d_model, rms_norm=rms_norm, dtype=dtype)
        self.dropout = dropout
        self.W_out = np.random.normal(0, 1.0 / np.sqrt(d_model), (d_model, vocab_size)).astype(dtype)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.opt_states = {}
        self.lr = 0.0
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.eps = 0.0
        self.opt_step = 0
        self.training = True

    def parameters(self) -> List[Tuple[str, np.ndarray]]:
        params = []
        params.append(('embed.W', self.embed.W))
        if not self.pos_embed.use_rotary:
            params.append(('pos.W', self.pos_embed.W))
        for i, layer in enumerate(self.layers):
            params.append((f'layer{i}.mha.W_q', layer.mha.W_q))
            params.append((f'layer{i}.mha.W_k', layer.mha.W_k))
            params.append((f'layer{i}.mha.W_v', layer.mha.W_v))
            params.append((f'layer{i}.mha.W_o', layer.mha.W_o))
            params.append((f'layer{i}.ln1.gamma', layer.ln1.gamma))
            params.append((f'layer{i}.ln1.beta', layer.ln1.beta))
            params.append((f'layer{i}.ff.W1', layer.ff.W1))
            params.append((f'layer{i}.ff.b1', layer.ff.b1))
            params.append((f'layer{i}.ff.W2', layer.ff.W2))
            params.append((f'layer{i}.ff.b2', layer.ff.b2))
            params.append((f'layer{i}.ln2.gamma', layer.ln2.gamma))
            params.append((f'layer{i}.ln2.beta', layer.ln2.beta))
            if layer.layer_scale:
                params.append((f'layer{i}.gamma1', layer.gamma1))
                params.append((f'layer{i}.gamma2', layer.gamma2))
        if not self.ln_f.rms_norm:
            params.append(('ln_f.gamma', self.ln_f.gamma))
            params.append(('ln_f.beta', self.ln_f.beta))
        else:
            params.append(('ln_f.weight', self.ln_f.weight))
        params.append(('W_out', self.W_out))
        return params

    def zero_grads(self):
        self.embed.grad_W.fill(0.0)
        if not self.pos_embed.use_rotary:
            self.pos_embed.grad_W.fill(0.0)
        for layer in self.layers:
            layer.mha.grad_W_q.fill(0.0)
            layer.mha.grad_W_k.fill(0.0)
            layer.mha.grad_W_v.fill(0.0)
            layer.mha.grad_W_o.fill(0.0)
            layer.ln1.grad_gamma.fill(0.0)
            layer.ln1.grad_beta.fill(0.0)
            layer.ff.grad_W1.fill(0.0)
            layer.ff.grad_b1.fill(0.0)
            layer.ff.grad_W2.fill(0.0)
            layer.ff.grad_b2.fill(0.0)
            layer.ln2.grad_gamma.fill(0.0)
            layer.ln2.grad_beta.fill(0.0)
        if not self.ln_f.rms_norm:
            self.ln_f.grad_gamma.fill(0.0)
            self.ln_f.grad_beta.fill(0.0)
        else:
            self.ln_f.grad_weight.fill(0.0)
        self.grad_W_out.fill(0.0)

    def forward(self, idx: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        b, s = idx.shape
        x = self.embed.forward(idx)
        if not self.pos_embed.use_rotary:
            x = x + self.pos_embed.forward(s)
        for layer in self.layers:
            x = layer.forward(x, training)
        x = self.ln_f.forward(x)
        if training and self.dropout > 0:
            dropout_mask = (np.random.rand(*x.shape) > self.dropout)
            x = x * dropout_mask / (1 - self.dropout)
        logits = x.reshape(-1, self.d_model) @ self.W_out
        logits = logits.reshape(b, s, -1)
        self._cache = {'x': x, 'idx': idx}
        return logits

    def loss_and_backward(self, idx_in: np.ndarray, idx_target: np.ndarray, 
                          grad_clip: float = 1.0) -> float:
        b, s = idx_in.shape
        logits = self.forward(idx_in, training=True)
        vocab = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = idx_target.reshape(-1)
        probs = softmax(logits_flat, axis=1)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])
        grad_logits = probs.copy()
        grad_logits[np.arange(grad_logits.shape[0]), targets_flat] -= 1
        grad_logits = grad_logits.reshape(b, s, vocab) / (b * s)
        x = self._cache['x']
        self.grad_W_out = x.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, vocab)
        dx = grad_logits.reshape(-1, vocab) @ self.W_out.T
        dx = dx.reshape(b, s, self.d_model)
        d_ln = self.ln_f.backward(dx)
        grad = d_ln
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        idx = self._cache['idx']
        self.embed.backward(idx, grad)
        if not self.pos_embed.use_rotary:
            self.pos_embed.backward(s, grad)
        if grad_clip > 0:
            total_norm = 0.0
            for _, param in self.parameters():
                if param.grad is not None:
                    param_norm = np.linalg.norm(param.grad)
                    total_norm += param_norm ** 2
            total_norm = np.sqrt(total_norm)
            clip_coef = min(grad_clip / (total_norm + EPS), 1.0)
            if clip_coef < 1:
                for _, param in self.parameters():
                    if param.grad is not None:
                        param.grad *= clip_coef
        return loss

    def init_optimizer(self, lr: float = 6e-4, betas=(0.9, 0.95), eps=1e-8, 
                       weight_decay: float = 0.1, warmup_steps: int = 2000):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.opt_step = 0
        self.opt_states = {}
        for name, param in self.parameters():
            self.opt_states[name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }

    def step_optimizer(self, current_step: Optional[int] = None):
        if current_step is not None:
            self.opt_step = current_step
        self.opt_step += 1
        if self.warmup_steps > 0:
            lr = self.lr * min(self.opt_step ** -0.5, self.opt_step * self.warmup_steps ** -1.5)
        else:
            lr = self.lr
        def update(name: str, param: np.ndarray, grad: np.ndarray):
            if 'W_' in name and self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            state = self.opt_states[name]
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)
            m_hat = state['m'] / (1 - self.beta1 ** self.opt_step)
            v_hat = state['v'] / (1 - self.beta2 ** self.opt_step)
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
        for name, param in self.parameters():
            if name in ['embed.W', 'pos.W', 'W_out'] or 'W_' in name:
                grad = getattr(self, f"grad_{name.split('.')[0]}")
            else:
                grad = getattr(self, f"grad_{name.replace('.', '_')}")
            update(name, param, grad)

    def enable_gradient_checkpointing(self):
        warnings.warn("Gradient checkpointing is not implemented in this NumPy version", RuntimeWarning)

    def convert_to_rms_norm(self):
        self.ln_f = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)
        for layer in self.layers:
            layer.ln1 = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)
            layer.ln2 = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)

    def save(self, path: str, include_optimizer: bool = False):
        data = {
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
            'embed.W': self.embed.W,
            'pos.W': self.pos_embed.W if not self.pos_embed.use_rotary else None,
            'layers': [],
            'ln_f.gamma': self.ln_f.gamma if not self.ln_f.rms_norm else None,
            'ln_f.beta': self.ln_f.beta if not self.ln_f.rms_norm else None,
            'ln_f.weight': self.ln_f.weight if self.ln_f.rms_norm else None,
            'W_out': self.W_out
        }
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
                'ln1.gamma': layer.ln1.gamma,
                'ln1.beta': layer.ln1.beta,
                'ln2.gamma': layer.ln2.gamma,
                'ln2.beta': layer.ln2.beta
            }
            if layer.layer_scale:
                layer_data['gamma1'] = layer.gamma1
                layer_data['gamma2'] = layer.gamma2
            data['layers'].append(layer_data)
        if include_optimizer and self.opt_states:
            data['optimizer'] = {
                'lr': self.lr,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'eps': self.eps,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'opt_step': self.opt_step,
                'states': {k: {'m': v['m'], 'v': v['v']} for k, v in self.opt_states.items()}
            }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, strict: bool = True):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embed.W = data['embed.W']
        if not self.pos_embed.use_rotary and data['pos.W'] is not None:
            self.pos_embed.W = data['pos.W']
        for layer, ld in zip(self.layers, data['layers']):
            layer.mha.W_q = ld['mha.W_q']
            layer.mha.W_k = ld['mha.W_k']
            layer.mha.W_v = ld['mha.W_v']
            layer.mha.W_o = ld['mha.W_o']
            layer.ff.W1 = ld['ff.W1']
            layer.ff.b1 = ld['ff.b1']
            layer.ff.W2 = ld['ff.W2']
            layer.ff.b2 = ld['ff.b2']
            layer.ln1.gamma = ld['ln1.gamma']
            layer.ln1.beta = ld['ln1.beta']
            layer.ln2.gamma = ld['ln2.gamma']
            layer.ln2.beta = ld['ln2.beta']
            if hasattr(layer, 'gamma1') and 'gamma1' in ld:
                layer.gamma1 = ld['gamma1']
            if hasattr(layer, 'gamma2') and 'gamma2' in ld:
                layer.gamma2 = ld['gamma2']
        if not self.ln_f.rms_norm:
            self.ln_f.gamma = data['ln_f.gamma']
            self.ln_f.beta = data['ln_f.beta']
        else:
            self.ln_f.weight = data['ln_f.weight']
        self.W_out = data['W_out']
        if 'optimizer' in data and self.opt_states:
            opt_data = data['optimizer']
            self.lr = opt_data['lr']
            self.beta1 = opt_data['beta1']
            self.beta2 = opt_data['beta2']
            self.eps = opt_data['eps']
            self.weight_decay = opt_data.get('weight_decay', 0.1)
            self.warmup_steps = opt_data.get('warmup_steps', 2000)
            self.opt_step = opt_data['opt_step']
            for name, state in opt_data['states'].items():
                if name in self.opt_states:
                    self.opt_states[name]['m'] = state['m']
                    self.opt_states[name]['v'] = state['v']

    def generate(self, idx_start: List[int], max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_k: Optional[int] = None, 
                 top_p: Optional[float] = None, do_sample: bool = True) -> List[int]:
        idx = list(idx_start)
        for _ in range(max_new_tokens):
            input_ids = np.array([idx[-self.max_len:]], dtype=np.int32)
            logits = self.forward(input_ids, training=False)
            next_logits = logits[0, -1] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                top_k = min(top_k, len(next_logits))
                top_k_idx = np.argpartition(next_logits, -top_k)[-top_k:]
                top_k_logits = next_logits[top_k_idx]
                if top_p is not None and top_p < 1.0:
                    sorted_idx = np.argsort(top_k_logits)[::-1]
                    sorted_logits = top_k_logits[sorted_idx]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    cutoff_idx = np.where(cumulative_probs > top_p)[0][0]
                    top_p_idx = top_k_idx[sorted_idx[:cutoff_idx + 1]]
                    top_p_logits = next_logits[top_p_idx]
                    probs = softmax(top_p_logits)
                    next_id = np.random.choice(top_p_idx, p=probs) if do_sample else top_p_idx[np.argmax(top_p_logits)]
                else:
                    probs = softmax(top_k_logits)
                    next_id = np.random.choice(top_k_idx, p=probs) if do_sample else top_k_idx[np.argmax(top_k_logits)]
            else:
                if top_p is not None and top_p < 1.0:
                    sorted_idx = np.argsort(next_logits)[::-1]
                    sorted_logits = next_logits[sorted_idx]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    cutoff_idx = np.where(cumulative_probs > top_p)[0][0]
                    top_p_idx = sorted_idx[:cutoff_idx + 1]
                    top_p_logits = next_logits[top_p_idx]
                    probs = softmax(top_p_logits)
                    next_id = np.random.choice(top_p_idx, p=probs) if do_sample else top_p_idx[np.argmax(top_p_logits)]
                else:
                    probs = softmax(next_logits)
                    next_id = np.random.choice(len(probs), p=probs) if do_sample else np.argmax(probs)
            idx.append(int(next_id))
        return idx

    def evaluate(self, val_data: np.ndarray, seq_len: int, batch_size: int, 
                 tokenizer: Any) -> Tuple[float, float]:
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        for xb, yb in get_batches_from_text(val_data, seq_len, batch_size, tokenizer):
            original_dropout = self.dropout
            self.dropout = 0.0
            b, s = xb.shape
            logits = self.forward(xb, training=False)
            vocab = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab)
            targets_flat = yb.reshape(-1)
            probs = softmax(logits_flat, axis=1)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])
            total_loss += loss * len(targets_flat)
            total_tokens += len(targets_flat)
            n_batches += 1
            self.dropout = original_dropout
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return avg_loss, perplexity

class Trainer:
    def __init__(self, model: GPT, tokenizer: Any, train_data: str, 
                 val_data: Optional[str] = None, seq_len: int = 1024, 
                 batch_size: int = 8, grad_accum_steps: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.history = {'train_loss': [], 'val_loss': [], 'perplexity': [], 'lr': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self, epochs: int = 10, lr: float = 3e-4, weight_decay: float = 0.1,
              warmup_steps: int = 2000, grad_clip: float = 1.0, 
              val_interval: int = 1, early_stopping_patience: int = 5,
              checkpoint_dir: str = 'checkpoints', save_best: bool = True):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.init_optimizer(
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps
        )
        total_steps = 0
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0
            total_steps += len(self.train_data) // (self.seq_len * self.batch_size)
            for i, (xb, yb) in enumerate(get_batches_from_text(
                    self.train_data, self.seq_len, self.batch_size, self.tokenizer)):
                loss = self.model.loss_and_backward(xb, yb, grad_clip)
                total_loss += loss
                n_batches += 1
                if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == n_batches:
                    self.model.step_optimizer(total_steps)
                    self.model.zero_grads()
                if i % 10 == 0:
                    current_lr = lr * min(total_steps ** -0.5, total_steps * warmup_steps ** -1.5) if warmup_steps > 0 else lr
                    print(f'Step {i+1}/{n_batches}, Loss: {loss:.4f}, LR: {current_lr:.2e}', end='\r')
            avg_loss = total_loss / max(1, n_batches)
            self.history['train_loss'].append(avg_loss)
            val_loss = float('inf')
            perplexity = float('inf')
            if self.val_data and epoch % val_interval == 0:
                val_loss, perplexity = self.model.evaluate(
                    self.val_data, self.seq_len, self.batch_size, self.tokenizer
                )
                self.history['val_loss'].append(val_loss)
                self.history['perplexity'].append(perplexity)
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(checkpoint_dir, 'best_model.pkl')
                    self.model.save(best_path, include_optimizer=True)
                    print(f"\n[INFO] Best model saved with validation loss: {val_loss:.4f}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f}")
            start_prompt = 'دوست '
            start_ids = [self.tokenizer.w2i.get(c, self.tokenizer.w2i['<unk>']) for c in start_prompt]
            gen = self.model.generate(start_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9)
            print('Sample:', self.tokenizer.decode(np.array(gen)))
            if epoch % 5 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pkl')
                self.model.save(ckpt_path)
                print(f"[INFO] Checkpoint saved to {ckpt_path}")
            if early_stopping_patience > 0 and self.patience_counter >= early_stopping_patience:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
                break
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        return self.history

if __name__ == '__main__':
    seq_len = 128
    batch_size = 8
    epochs = 50
    lr = 6e-4
    try:
        with open('sample_text.txt', 'r', encoding='utf-8') as f:
            sample_text = f.read()
    except:
        sample_text = """
        دوست دارم برنامه‌نویسی کنم. این یک متن نمونه است برای آموزش مدل GPT کوچک.
        مدل می‌تواند کاراکترها را یاد بگیرد و متن تولید کند.
        هوش مصنوعی یکی از حوزه‌های پررونق در دنیای امروز است.
        مدل‌های زبانی بزرگ قادر به انجام کارهای شگفت‌انگیزی هستند.
        در این مثال ساده، ما یک مدل GPT کوچک را پیاده‌سازی می‌کنیم.
        """
    train_ratio = 0.9
    split_idx = int(len(sample_text) * train_ratio)
    train_text = sample_text[:split_idx]
    val_text = sample_text[split_idx:]
    print("Building tokenizer...")
    tok = BPETokenizer()
    tok.build_from_text([train_text], vocab_size=500)
    vocab_size = len(tok.vocab)
    print(f'Vocabulary size: {vocab_size}')
    print("Building model...")
    model = GPT(
        vocab_size=vocab_size,
        max_len=seq_len,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=6,
        dropout=0.1,
        use_rotary=False,
        rms_norm=True,
        layer_scale=True
    )
    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        tokenizer=tok,
        train_data=train_text,
        val_data=val_text,
        seq_len=seq_len,
        batch_size=batch_size
    )
    history = trainer.train(
        epochs=epochs,
        lr=lr,
        weight_decay=0.1,
        warmup_steps=1000,
        grad_clip=1.0,
        val_interval=1,
        early_stopping_patience=10,
        checkpoint_dir='checkpoints'
    )
    model.save('gpt_final.pkl')
    print('Final model saved -> gpt_final.pkl')







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
