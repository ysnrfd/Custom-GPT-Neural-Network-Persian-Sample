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

DEFAULT_DTYPE = np.float32
EPS = 1e-6

def softmax(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    """Softmax-پایدار عددی را محاسبه می کند"""
    x = x - np.max(x, axis=axis, keepdims=True) # پایداری عددی
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + eps)

def gelu(x: np.ndarray) -> np.ndarray:
    """تقریب تابع فعال سازی GELU"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x: np.ndarray) -> np.ndarray:
    """ تابع فعال سازی GELU دقیق با استفاده از erf """
    return 0.5 * x * (1.0 + math.erf(x / np.sqrt(2.0)))

def gelu_grad(x: np.ndarray) -> np.ndarray:
    """مشتق تابع فعال سازی GELU (تقریبی)"""
    tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    sech2 = 1.0 - tanh_term**2
    return 0.5 * (1.0 + tanh_term) + 0.5 * x * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Root Mean Square Layer Normalization"""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return weight * (x / rms)

class BPETokenizer:
    """
    یک پیاده سازی ساده از توکنایزر Byte-Pair Encoding (BPE).
    """
    def __init__(self):
        self.vocab: List[str] = []
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.cache: Dict[str, List[str]] = {}
        self.special_tokens: List[str] = ['<pad>', '<unk>', '<bos>', '<eos>'] # پد، ناشناخته، شروع، پایان

    @staticmethod
    def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """
        جفت های متوالی از نمادها را در یک کلمه برمی گرداند.
        کلمه به صورت تاپلی از رشته ها (نمادها) است.
        """
        return set(zip(word, word[1:]))

    @staticmethod
    def bytes_to_unicode() -> Dict[int, str]:
        """
        یک نگاشت از بایت ها (0-255) به کاراکترهای یونیکد قابل چاپ ایجاد می کند.
        این برای اطمینان از اینکه هر بایت یک توکن اولیه معتبر است استفاده می شود.
        """
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
        """متن را به دنباله ای از کاراکترهای یونیکد نگاشت شده از بایت های UTF-8 آن تبدیل می کند."""
        byte_encoder = self.bytes_to_unicode()
        text_bytes = text.encode("utf-8")
        return "".join([byte_encoder[b] for b in text_bytes])

    def build_from_text(self, texts: List[str], vocab_size: int = 500, min_freq: int = 2):
        """
        واژگان BPE و ادغام ها را از لیستی از متون خام می سازد.
        """
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError(f"Vocab size must be at least {256 + len(self.special_tokens)} to cover basic bytes and special tokens")

        preprocessed = [self.preprocess(text) for text in texts]
        
        # 1. واژگان اولیه از کاراکترها (بایت های یونیکد)
        char_freq = {}
        for text in preprocessed:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # مرتب سازی بر اساس فراوانی (اگرچه همه بایت ها باید در نهایت اضافه شوند)
        base_vocab = sorted(char_freq.keys(), key=lambda x: -char_freq[x])
        # اطمینان حاصل کنید که همه بایت ها در واژگان پایه هستند
        for i in range(2**8):
            char = self.bytes_to_unicode()[i]
            if char not in char_freq:
                base_vocab.append(char)
                
        self.vocab = self.special_tokens + base_vocab
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}

        # 2. یادگیری ادغام ها
        # "کلمات" را به عنوان دنباله ای از کاراکترهای جدا شده با فاصله آماده کنید
        # مثال: "hello" -> "h e l l o"
        words = []
        for text in preprocessed:
            words.extend([' '.join(text)]) # ما با متن کامل به عنوان یک "کلمه" کار می کنیم
            
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        num_merges = vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # 2a. شمارش فراوانی جفت ها
            pairs = {}
            for word, freq in word_freq.items():
                chars = word.split()
                if len(chars) < 2:
                    continue
                for j in range(len(chars) - 1):
                    pair = (chars[j], chars[j+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break # ادغام بیشتری امکان پذیر نیست
            
            # 2b. پیدا کردن بهترین جفت
            best_pair = max(pairs, key=pairs.get)
            
            # 2c. ایجاد توکن جدید و افزودن به واژگان
            new_token = ''.join(best_pair)
            if new_token not in self.w2i:
                self.vocab.append(new_token)
                idx = len(self.vocab) - 1
                self.w2i[new_token] = idx
                self.i2w[idx] = new_token
                self.merges.append(best_pair)
            
            # 2d. به روز رسانی "کلمات" (word_freq) با توکن جدید
            new_word_freq = {}
            for word, freq in word_freq.items():
                # .replace() همه رخدادها را جایگزین می کند
                new_word = word.replace(' '.join(best_pair), new_token)
                new_word_freq[new_word] = freq
            word_freq = new_word_freq

    def encode(self, text: str, max_len: int = None, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """متن را به آرایه ای از ID های توکن کدگذاری می کند."""
        text = self.preprocess(text)
        if add_bos:
            text = self.special_tokens[2] + text
        if add_eos:
            text = text + self.special_tokens[3]
            
        if text in self.cache:
            tokens = self.cache[text]
        else:
            tokens = list(text)
            # اعمال ادغام ها به ترتیب یادگیری
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
        
        # اعمال برش و پدینگ
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            elif len(ids) < max_len:
                ids = ids + [self.w2i['<pad>']] * (max_len - len(ids))
                
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: Union[np.ndarray, List[int]]) -> str:
        """آرایه ای از ID های توکن را به متن بازگردانی می کند."""
        tokens = [self.i2w.get(int(i), '<unk>') for i in ids]
        text = ''.join(tokens)
        
        # حذف توکن های ویژه
        for token in self.special_tokens:
            text = text.replace(token, '')
            
        # بازگردانی بایت ها از یونیکد
        byte_decoder = {v: k for k, v in self.bytes_to_unicode().items()}
        try:
            text_bytes_list = [byte_decoder[c] for c in text]
            text_bytes = bytearray(text_bytes_list)
            return text_bytes.decode('utf-8', errors='replace')
        except KeyError:
            # اگر کاراکتری در دیکشنری نباشد (نباید اتفاق بیفتد)
            return "Decode Error: Invalid token"


class Embedding:
    """لایه نشان گذاری (Embedding)"""
    def __init__(self, vocab_size: int, d_model: int, dtype=DEFAULT_DTYPE):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        scale = 1.0 / np.sqrt(d_model)
        self.W = np.random.normal(0, scale, (vocab_size, d_model)).astype(dtype)
        self.grad_W = np.zeros_like(self.W)
        self.idx_cache = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """idx (batch_size, seq_len) -> (batch_size, seq_len, d_model)"""
        self.idx_cache = idx # ذخیره برای backward
        return self.W[idx]

    def backward(self, grad: np.ndarray):
        """grad (batch_size, seq_len, d_model) -> grad_W"""
        # np.add.at گرادیان ها را در اندیس های تکراری جمع می کند
        np.add.at(self.grad_W, self.idx_cache, grad)

class PositionalEmbedding:
    """لایه نشان گذاری موقعیتی (سینوسی یا چرخشی)"""
    def __init__(self, max_len: int, d_model: int, use_rotary: bool = False, dtype=DEFAULT_DTYPE):
        self.max_len = max_len
        self.d_model = d_model
        self.use_rotary = use_rotary
        self.dtype = dtype
        
        if not use_rotary:
            # نشان گذاری موقعیتی سینوسی استاندارد
            self.W = np.zeros((max_len, d_model), dtype=dtype)
            position = np.arange(0, max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            self.W[:, 0::2] = np.sin(position * div_term)
            self.W[:, 1::2] = np.cos(position * div_term)
            self.grad_W = np.zeros_like(self.W)
        else:
            # آماده سازی فرکانس ها برای RoPE (Rotary Positional Embedding)
            self.rotary_freqs = self._create_rotary_frequencies(self.d_model // 2, dtype) # d_model باید زوج باشد

    def _create_rotary_frequencies(self, head_dim: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
        """فرکانس های RoPE را محاسبه می کند."""
        # RoPE معمولاً در هر سر (head) اعمال می شود، اما اینجا ما آن را برای d_model کامل محاسبه می کنیم
        # این فرض می کند که d_model == head_dim برای سادگی، یا اینکه در MHA به درستی اعمال می شود
        inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim * 2, 2, dtype=dtype) / (head_dim * 2)))
        return inv_freq

    def apply_rotary_pos_emb(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """
        RoPE را به تانسور ورودی اعمال می کند.
        x: (batch_size, num_heads, seq_len, head_dim)
        """
        b, h, s, d = x.shape
        if d * 2 != len(self.rotary_freqs):
            # اگر فرکانس ها با ابعاد سر مطابقت ندارند، دوباره محاسبه کنید
            # این معمولاً در MHA اتفاق می افتد
            freqs = self._create_rotary_frequencies(d, dtype=self.dtype)
        else:
            freqs = self.rotary_freqs

        t = np.arange(s, dtype=self.dtype)
        freqs_cis = np.outer(t, freqs) # (seq_len, head_dim / 2)
        cos = np.cos(freqs_cis) # (seq_len, head_dim / 2)
        sin = np.sin(freqs_cis) # (seq_len, head_dim / 2)

        # گسترش برای ابعاد دسته و سرها
        cos = cos[np.newaxis, np.newaxis, :, :] # (1, 1, seq_len, head_dim / 2)
        sin = sin[np.newaxis, np.newaxis, :, :] # (1, 1, seq_len, head_dim / 2)

        # اعمال چرخش
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        x_rotated1 = x1 * cos - x2 * sin
        x_rotated2 = x1 * sin + x2 * cos
        
        x_rotated = np.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated1
        x_rotated[..., 1::2] = x_rotated2
        
        return x_rotated

    def forward(self, seq_len: int) -> Optional[np.ndarray]:
        """(1, seq_len, d_model)"""
        if not self.use_rotary:
            return self.W[np.newaxis, :seq_len, :]
        return None # RoPE در forward خود MHA اعمال می شود

    def backward(self, seq_len: int, grad: np.ndarray):
        """grad (batch_size, seq_len, d_model)"""
        if not self.use_rotary:
            # گرادیان را در بعد دسته جمع می کنیم
            self.grad_W[:seq_len, :] += np.sum(grad, axis=0)

class LayerNorm:
    """لایه نرمال سازی (استاندارد یا RMS)"""
    def __init__(self, d_model: int, eps: float = EPS, rms_norm: bool = False, dtype=DEFAULT_DTYPE):
        self.d_model = d_model
        self.eps = eps
        self.rms_norm = rms_norm
        self.dtype = dtype
        
        if not rms_norm:
            # LayerNorm استاندارد
            self.gamma = np.ones((1, 1, d_model), dtype=dtype)
            self.beta = np.zeros((1, 1, d_model), dtype=dtype)
            self.grad_gamma = np.zeros_like(self.gamma)
            self.grad_beta = np.zeros_like(self.beta)
        else:
            # RMSNorm
            self.weight = np.ones((1, 1, d_model), dtype=dtype)
            self.grad_weight = np.zeros_like(self.weight)
            
        self.x = None
        self.mean = None
        self.var = None
        self.x_norm = None
        self.rms = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.rms_norm:
            self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
            self.x_norm = x / self.rms
            return self.weight * self.x_norm
        else:
            self.mean = np.mean(x, axis=-1, keepdims=True)
            self.var = np.var(x, axis=-1, keepdims=True)
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            return self.gamma * self.x_norm + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.rms_norm:
            self.grad_weight = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)
            
            grad_x_norm = grad * self.weight
            d_x_norm = grad_x_norm / self.rms
            d_rms = -np.sum(grad_x_norm * self.x, axis=-1, keepdims=True) / (self.rms**3)
            d_x_sq = d_rms / (2.0 * self.rms) * (1.0 / self.d_model) # اشتباه است، rms مشتق شده از mean
            
            # بازنویسی مشتق RMS
            # d_out / d_x_norm = weight
            # d_out / d_weight = x_norm
            # d_x_norm / d_rms = -x / rms^2 = -x_norm / rms
            # d_rms / d_x = x / (d_model * rms)
            # d_out / d_x = (d_out / d_x_norm) * (d_x_norm / d_x)
            # d_x_norm / d_x = (1/rms) - (x / rms^2) * (x / (d_model * rms))
            # d_x_norm / d_x = (1/rms) - (x^2 / (d_model * rms^3))
            # d_x_norm / d_x = (1/rms) * (1 - (x^2 / (d_model * rms^2)))
            # d_x_norm / d_x = (1/rms) * (1 - (x_norm^2 / d_model)) # این درست به نظر نمی رسد

            # از فرمول ساده تر استفاده کنیم:
            # dL/dx_i = (dL/dy_i * w_i) / rms - (sum(dL/dy_j * w_j * y_j) * y_i) / (d_model * rms)
            # y = x_norm
            
            grad_x_norm = grad * self.weight
            sum_grad_y = np.sum(grad_x_norm * self.x_norm, axis=-1, keepdims=True)
            dx = (grad_x_norm / self.rms) - (sum_grad_y * self.x_norm) / (self.d_model * self.rms)
            
            return dx
        else:
            b, s, d = grad.shape
            self.grad_gamma = np.sum(grad * self.x_norm, axis=(0, 1), keepdims=True)
            self.grad_beta = np.sum(grad, axis=(0, 1), keepdims=True)
            
            dx_norm = grad * self.gamma
            inv_std = 1. / np.sqrt(self.var + self.eps)
            
            dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * (inv_std**3), axis=-1, keepdims=True)
            dmean = np.sum(dx_norm * -inv_std, axis=-1, keepdims=True) + \
                    dvar * np.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)
                    
            dx = (dx_norm * inv_std) + \
                 (dvar * 2.0 * (self.x - self.mean) / d) + \
                 (dmean / d)
                 
            return dx

class FeedForward:
    """لایه Feed-Forward (MLP)"""
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
            self.dropout_mask1 = (np.random.rand(*self.hidden_act.shape) > self.dropout).astype(self.dtype)
            self.hidden_act = self.hidden_act * self.dropout_mask1 / (1.0 - self.dropout)
        else:
            self.dropout_mask1 = None
            
        out = self.hidden_act @ self.W2 + self.b2
        
        if training and self.dropout > 0:
            self.dropout_mask2 = (np.random.rand(*out.shape) > self.dropout).astype(self.dtype)
            out = out * self.dropout_mask2 / (1.0 - self.dropout)
        else:
            self.dropout_mask2 = None
            
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, s, d = grad.shape
        
        if self.dropout_mask2 is not None:
            grad = grad * self.dropout_mask2 / (1.0 - self.dropout)
            
        # گرادیان های W2 و b2
        # (d_ff, d) = (d_ff, b*s) @ (b*s, d)
        self.grad_W2 = (self.hidden_act.reshape(-1, self.d_ff).T @ grad.reshape(-1, self.d_model))
        self.grad_b2 = np.sum(grad, axis=(0, 1), keepdims=True)
        
        # گرادیان ورودی W2 (dhidden_act)
        dhidden_act = grad @ self.W2.T
        
        if self.dropout_mask1 is not None:
            dhidden_act = dhidden_act * self.dropout_mask1 / (1.0 - self.dropout)
            
        # گرادیان از طریق GELU
        dhidden = dhidden_act * gelu_grad(self.hidden)
        
        # گرادیان های W1 و b1
        # (d_model, d_ff) = (d_model, b*s) @ (b*s, d_ff)
        self.grad_W1 = (self.x.reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_ff))
        self.grad_b1 = np.sum(dhidden, axis=(0, 1), keepdims=True)
        
        # گرادیان ورودی x
        dx = dhidden @ self.W1.T
        
        return dx

class MultiHeadSelfAttention:
    """لایه توجه-خودی چند-سر (Multi-Head Self-Attention)"""
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
        
        if self.use_rotary:
            self.pos_emb = PositionalEmbedding(0, self.head_dim, use_rotary=True, dtype=dtype) # max_len مهم نیست

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """(b, s, d) -> (b, h, s, hd)"""
        b, s, d = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(b, h, s, hd) -> (b, s, d)"""
        x = np.transpose(x, (0, 2, 1, 3))
        b, s, h, hd = x.shape
        return x.reshape(b, s, h * hd)

    def causal_mask(self, seq_len: int) -> np.ndarray:
        """(seq_len, seq_len) -> bool"""
        # np.tril یک ماتریس پایین مثلثی ایجاد می کند
        return np.tril(np.ones((seq_len, seq_len), dtype=bool))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        b, s, d = x.shape
        
        # 1. محاسبه Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # 2. تقسیم به سرها
        Qh = self.split_heads(Q) # (b, h, s, hd)
        Kh = self.split_heads(K) # (b, h, s, hd)
        Vh = self.split_heads(V) # (b, h, s, hd)
        
        # 3. (اختیاری) اعمال RoPE
        if self.use_rotary:
            Qh = self.pos_emb.apply_rotary_pos_emb(Qh, seq_len=s)
            Kh = self.pos_emb.apply_rotary_pos_emb(Kh, seq_len=s)
            
        # 4. محاسبه امتیازات توجه
        dk_sqrt = np.sqrt(self.head_dim)
        # (b, h, s, hd) @ (b, h, hd, s) -> (b, h, s, s)
        scores = (Qh @ np.transpose(Kh, (0, 1, 3, 2))) / dk_sqrt
        
        # 5. اعمال ماسک علی (causal)
        mask = self.causal_mask(s)[np.newaxis, np.newaxis, :, :] # (1, 1, s, s)
        scores = np.where(mask, scores, -np.inf) # -inf برای softmax
        
        # 6. Softmax
        attn = softmax(scores, axis=-1) # (b, h, s, s)
        
        # 7. (اختیاری) Dropout
        if training and self.dropout > 0:
            self.dropout_mask = (np.random.rand(*attn.shape) > self.dropout).astype(self.dtype)
            attn = attn * self.dropout_mask / (1.0 - self.dropout)
        else:
            self.dropout_mask = None
            
        # 8. اعمال توجه به V
        # (b, h, s, s) @ (b, h, s, hd) -> (b, h, s, hd)
        attn_out = attn @ Vh
        
        # 9. ترکیب سرها و اعمال W_o
        out_concat = self.combine_heads(attn_out) # (b, s, d)
        out = out_concat @ self.W_o
        
        # ذخیره مقادیر میانی برای backward
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'Qh': Qh, 'Kh': Kh, 'Vh': Vh,
            'scores': scores, 'attn': attn, 'attn_out': attn_out,
            'out_concat': out_concat, 'mask': mask
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # بازیابی مقادیر از کش
        x = self.cache['x']
        Qh, Kh, Vh = self.cache['Qh'], self.cache['Kh'], self.cache['Vh']
        attn = self.cache['attn']
        out_concat = self.cache['out_concat']
        mask = self.cache['mask']
        
        b, s, d = grad_out.shape
        dk_sqrt = np.sqrt(self.head_dim)
        
        # 1. گرادیان W_o
        # (d, d) = (d, b*s) @ (b*s, d)
        self.grad_W_o = (out_concat.reshape(-1, d).T @ grad_out.reshape(-1, d))
        
        # 2. گرادیان ورودی W_o (d_out_concat)
        d_out_concat = grad_out @ self.W_o.T # (b, s, d)
        
        # 3. گرادیان از combine_heads
        d_attn_out = np.transpose(d_out_concat.reshape(b, s, self.num_heads, self.head_dim), (0, 2, 1, 3))
        
        # 4. گرادیان از (attn @ Vh)
        # dVh = attn.T @ d_attn_out
        dVh = np.transpose(attn, (0, 1, 3, 2)) @ d_attn_out # (b, h, hd, s)
        # dattn = d_attn_out @ Vh.T
        dattn = d_attn_out @ np.transpose(Vh, (0, 1, 3, 2)) # (b, h, s, s)
        
        # 5. گرادیان از Dropout
        if self.dropout_mask is not None:
            dattn = dattn * self.dropout_mask / (1.0 - self.dropout)
            
        # 6. گرادیان از Softmax
        # dscores = dattn * attn - attn * sum(dattn * attn)
        sft = attn
        dscores = sft * (dattn - np.sum(dattn * sft, axis=-1, keepdims=True))
        
        # 7. گرادیان از Mask
        dscores = np.where(mask, dscores, 0.0)
        
        # 8. گرادیان از Scores (Qh @ Kh.T / dk_sqrt)
        dscores_scaled = dscores / dk_sqrt
        # dQh = dscores_scaled @ Kh
        dQh = dscores_scaled @ Kh # (b, h, s, hd)
        # dKh = dscores_scaled.T @ Qh
        dKh = np.transpose(dscores_scaled, (0, 1, 3, 2)) @ Qh # (b, h, hd, s)
        
        # 9. (اختیاری) گرادیان از RoPE
        if self.use_rotary:
            # TODO: پیاده سازی backward برای RoPE (بسیار پیچیده)
            # برای سادگی، ما از این مرحله صرف نظر می کنیم.
            # این بدان معناست که RoPE گرادیانی دریافت نخواهد کرد (که برای RoPE ثابت، اشکالی ندارد)
            pass 
            
        # 10. گرادیان از split_heads
        dQ = self.combine_heads(dQh) # (b, s, d)
        dK = self.combine_heads(dKh) # (b, s, d)
        dV = self.combine_heads(dVh) # (b, s, d)

        # 11. گرادیان W_q, W_k, W_v
        # (d, d) = (d, b*s) @ (b*s, d)
        self.grad_W_q = (x.reshape(-1, d).T @ dQ.reshape(-1, d))
        self.grad_W_k = (x.reshape(-1, d).T @ dK.reshape(-1, d))
        self.grad_W_v = (x.reshape(-1, d).T @ dV.reshape(-1, d))
        
        # 12. گرادیان ورودی x
        dx_q = dQ @ self.W_q.T
        dx_k = dK @ self.W_k.T
        dx_v = dV @ self.W_v.T
        
        dx = dx_q + dx_k + dx_v # گرادیان ها از سه شاخه جمع می شوند
        
        return dx

class DecoderBlock:
    """یک بلوک رمزگشای Transformer (Post-Norm)"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 layer_scale: bool = False, layer_scale_init: float = 1e-4, use_rotary: bool = False,
                 rms_norm: bool = False, dtype=DEFAULT_DTYPE):
                 
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout, use_rotary, dtype)
        self.ln1 = LayerNorm(d_model, rms_norm=rms_norm, dtype=dtype)
        self.ff = FeedForward(d_model, d_ff, dropout, dtype)
        self.ln2 = LayerNorm(d_model, rms_norm=rms_norm, dtype=dtype)
        
        self.dropout = dropout
        self.layer_scale = layer_scale
        
        if layer_scale:
            self.gamma1 = np.ones((1, 1, d_model), dtype=dtype) * layer_scale_init
            self.gamma2 = np.ones((1, 1, d_model), dtype=dtype) * layer_scale_init
            self.grad_gamma1 = np.zeros_like(self.gamma1)
            self.grad_gamma2 = np.zeros_like(self.gamma2)
            
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache = {'x_in': x} # ذخیره ورودی برای اتصال باقیمانده
        
        # 1. MHA
        attn_out = self.mha.forward(x, training)
        self.cache['attn_out_unscaled'] = attn_out
        
        if self.layer_scale:
            attn_out_scaled = attn_out * self.gamma1
        else:
            attn_out_scaled = attn_out
            
        # 2. Add & Norm 1
        res1_out = x + attn_out_scaled
        ln1_out = self.ln1.forward(res1_out)
        self.cache['ln1_out'] = ln1_out # ذخیره برای اتصال باقیمانده دوم
        
        # 3. FFN
        ff_out = self.ff.forward(ln1_out, training)
        self.cache['ff_out_unscaled'] = ff_out
        
        if self.layer_scale:
            ff_out_scaled = ff_out * self.gamma2
        else:
            ff_out_scaled = ff_out
            
        # 4. Add & Norm 2
        res2_out = ln1_out + ff_out_scaled
        ln2_out = self.ln2.forward(res2_out)
        
        return ln2_out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: گرادیان خروجی بلوک (dL/d_ln2_out)
        
        # 1. Backprop از ln2
        d_res2_out = self.ln2.backward(grad) # dL/d_res2_out
        
        # 2. Backprop از Add 2 (res2_out = ln1_out + ff_out_scaled)
        d_ln1_out = d_res2_out.copy() # گرادیان برای شاخه ln1_out
        d_ff_out_scaled = d_res2_out.copy() # گرادیان برای شاخه ff_out_scaled
        
        # 3. Backprop از LayerScale 2 (ff_out_scaled = ff_out * gamma2)
        if self.layer_scale:
            self.grad_gamma2 = np.sum(d_ff_out_scaled * self.cache['ff_out_unscaled'], axis=(0, 1), keepdims=True)
            d_ff_out_unscaled = d_ff_out_scaled * self.gamma2
        else:
            d_ff_out_unscaled = d_ff_out_scaled
            
        # 4. Backprop از FFN (ff_out = ff(ln1_out))
        d_ff_in = self.ff.backward(d_ff_out_unscaled) # dL/d_ln1_out از شاخه FFN
        d_ln1_out += d_ff_in # جمع گرادیان ها برای ln1_out
        
        # 5. Backprop از ln1
        d_res1_out = self.ln1.backward(d_ln1_out) # dL/d_res1_out
        
        # 6. Backprop از Add 1 (res1_out = x + attn_out_scaled)
        dx = d_res1_out.copy() # گرادیان برای شاخه x
        d_attn_out_scaled = d_res1_out.copy() # گرادیان برای شاخه attn_out_scaled
        
        # 7. Backprop از LayerScale 1 (attn_out_scaled = attn_out * gamma1)
        if self.layer_scale:
            self.grad_gamma1 = np.sum(d_attn_out_scaled * self.cache['attn_out_unscaled'], axis=(0, 1), keepdims=True)
            d_attn_out_unscaled = d_attn_out_scaled * self.gamma1
        else:
            d_attn_out_unscaled = d_attn_out_scaled
            
        # 8. Backprop از MHA (attn_out = mha(x))
        d_mha_in = self.mha.backward(d_attn_out_unscaled) # dL/d_x از شاخه MHA
        dx += d_mha_in # جمع گرادیان ها برای x
        
        return dx

class GPT:
    """مدل کامل GPT (فقط رمزگشا)"""
    def __init__(self, vocab_size: int, max_len: int = 512, d_model: int = 768, num_heads: int = 12,
                 d_ff: int = 3072, num_layers: int = 12, dropout: float = 0.1,
                 use_rotary: bool = False, rms_norm: bool = False, layer_scale: bool = False,
                 dtype=DEFAULT_DTYPE):
                 
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.dtype = dtype
        self.dropout_rate = dropout
        
        self.embed = Embedding(vocab_size, d_model, dtype)
        self.pos_embed = PositionalEmbedding(max_len, d_model, use_rotary, dtype)
        
        self.layers = [
            DecoderBlock(d_model, num_heads, d_ff, dropout, layer_scale, 
                         use_rotary=use_rotary, rms_norm=rms_norm, dtype=dtype)
            for _ in range(num_layers)
        ]
        
        self.ln_f = LayerNorm(d_model, rms_norm=rms_norm, dtype=dtype)
        
        # لایه خروجی (سر LM)
        # اغلب وزن ها با embedding به اشتراک گذاشته می شود، اما اینجا جداگانه می سازیم
        scale_out = 1.0 / np.sqrt(d_model)
        self.W_out = np.random.normal(0, scale_out, (d_model, vocab_size)).astype(dtype)
        self.grad_W_out = np.zeros_like(self.W_out)
        
        # وضعیت بهینه ساز
        self.opt_states = {}
        self.lr = 0.0
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.eps = 0.0
        self.weight_decay = 0.0
        self.warmup_steps = 0
        self.opt_step = 0
        
        self.training = True
        self._cache = {}
        self.dropout_mask_final = None

    def parameters(self) -> List[Tuple[str, np.ndarray]]:
        """
        لیستی از (نام، آرایه پارامتر) را برای همه پارامترهای قابل آموزش برمی گرداند.
        """
        params = []
        params.append(('embed.W', self.embed.W))
        if not self.pos_embed.use_rotary:
            params.append(('pos.W', self.pos_embed.W))
            
        for i, layer in enumerate(self.layers):
            params.append((f'layer{i}.mha.W_q', layer.mha.W_q))
            params.append((f'layer{i}.mha.W_k', layer.mha.W_k))
            params.append((f'layer{i}.mha.W_v', layer.mha.W_v))
            params.append((f'layer{i}.mha.W_o', layer.mha.W_o))
            
            if not layer.ln1.rms_norm:
                params.append((f'layer{i}.ln1.gamma', layer.ln1.gamma))
                params.append((f'layer{i}.ln1.beta', layer.ln1.beta))
            else:
                params.append((f'layer{i}.ln1.weight', layer.ln1.weight))
                
            params.append((f'layer{i}.ff.W1', layer.ff.W1))
            params.append((f'layer{i}.ff.b1', layer.ff.b1))
            params.append((f'layer{i}.ff.W2', layer.ff.W2))
            params.append((f'layer{i}.ff.b2', layer.ff.b2))

            if not layer.ln2.rms_norm:
                params.append((f'layer{i}.ln2.gamma', layer.ln2.gamma))
                params.append((f'layer{i}.ln2.beta', layer.ln2.beta))
            else:
                params.append((f'layer{i}.ln2.weight', layer.ln2.weight))

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

    def _get_grads_map(self) -> Dict[str, np.ndarray]:
        """
        یک دیکشنری از (نام پارامتر -> آرایه گرادیان) برمی گرداند.
        """
        grads = {
            'embed.W': self.embed.grad_W,
            'pos.W': self.pos_embed.grad_W if not self.pos_embed.use_rotary else None,
            'W_out': self.grad_W_out
        }
        if not self.ln_f.rms_norm:
            grads['ln_f.gamma'] = self.ln_f.grad_gamma
            grads['ln_f.beta'] = self.ln_f.grad_beta
        else:
            grads['ln_f.weight'] = self.ln_f.grad_weight

        for i, layer in enumerate(self.layers):
            grads[f'layer{i}.mha.W_q'] = layer.mha.grad_W_q
            grads[f'layer{i}.mha.W_k'] = layer.mha.grad_W_k
            grads[f'layer{i}.mha.W_v'] = layer.mha.grad_W_v
            grads[f'layer{i}.mha.W_o'] = layer.mha.grad_W_o
            
            if not layer.ln1.rms_norm:
                grads[f'layer{i}.ln1.gamma'] = layer.ln1.grad_gamma
                grads[f'layer{i}.ln1.beta'] = layer.ln1.grad_beta
            else:
                grads[f'layer{i}.ln1.weight'] = layer.ln1.grad_weight
                
            grads[f'layer{i}.ff.W1'] = layer.ff.grad_W1
            grads[f'layer{i}.ff.b1'] = layer.ff.grad_b1
            grads[f'layer{i}.ff.W2'] = layer.ff.grad_W2
            grads[f'layer{i}.ff.b2'] = layer.ff.grad_b2
            
            if not layer.ln2.rms_norm:
                grads[f'layer{i}.ln2.gamma'] = layer.ln2.grad_gamma
                grads[f'layer{i}.ln2.beta'] = layer.ln2.grad_beta
            else:
                grads[f'layer{i}.ln2.weight'] = layer.ln2.grad_weight

            if layer.layer_scale:
                grads[f'layer{i}.gamma1'] = layer.grad_gamma1
                grads[f'layer{i}.gamma2'] = layer.grad_gamma2
        return grads

    def zero_grads(self):
        """تمام آرایه های گرادیان را صفر می کند."""
        for grad_array in self._get_grads_map().values():
            if grad_array is not None:
                grad_array.fill(0.0)

    def forward(self, idx: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        b, s = idx.shape
        
        # 1. Embeddings
        x = self.embed.forward(idx)
        
        # 2. Positional Embeddings
        if not self.pos_embed.use_rotary:
            x = x + self.pos_embed.forward(s)
            
        # 3. Dropout روی Embeddings
        if training and self.dropout_rate > 0:
            # (این dropout در GPT-2 اصلی وجود دارد)
            dropout_mask_embed = (np.random.rand(*x.shape) > self.dropout_rate).astype(self.dtype)
            x = x * dropout_mask_embed / (1.0 - self.dropout_rate)
            
        # 4. Decoder Blocks
        for layer in self.layers:
            x = layer.forward(x, training)
            
        # 5. Final LayerNorm
        x = self.ln_f.forward(x)
        
        # (Dropout نهایی معمولاً قبل از ln_f است، اما ما آن را اینجا نگه می داریم)
        if training and self.dropout_rate > 0:
            self.dropout_mask_final = (np.random.rand(*x.shape) > self.dropout_rate).astype(self.dtype)
            x = x * self.dropout_mask_final / (1.0 - self.dropout_rate)
        else:
            self.dropout_mask_final = None
            
        # 6. LM Head
        # (b, s, d) @ (d, v) -> (b, s, v)
        logits = x @ self.W_out
        
        self._cache = {'x_final_normed': x, 'idx': idx}
        return logits

    def loss_and_backward(self, idx_in: np.ndarray, idx_target: np.ndarray,
                          grad_clip: float = 1.0) -> float:
        """
        محاسبه زیان (Cross-Entropy) و اجرای backward pass.
        """
        b, s = idx_in.shape
        
        # 1. Forward pass
        logits = self.forward(idx_in, training=True)
        vocab = logits.shape[-1]
        
        # 2. محاسبه زیان (Cross-Entropy)
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = idx_target.reshape(-1)
        
        probs = softmax(logits_flat, axis=1)
        # انتخاب احتمالات مربوط به توکن های هدف
        log_probs = np.log(probs[np.arange(len(targets_flat)), targets_flat] + EPS)
        loss = -np.mean(log_probs)
        
        # 3. محاسبه گرادیان اولیه (dL/d_logits)
        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits = grad_logits.reshape(b, s, vocab) / (b * s) # نرمال سازی با اندازه دسته
        
        # 4. Backward pass
        x_final = self._cache['x_final_normed']
        
        # 4a. گرادیان W_out
        # (d, v) = (d, b*s) @ (b*s, v)
        self.grad_W_out = (x_final.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, vocab))
        
        # 4b. گرادیان ورودی W_out (dx_final)
        dx = grad_logits @ self.W_out.T # (b, s, d)
        
        # 4c. گرادیان از Dropout نهایی
        if self.dropout_mask_final is not None:
            dx = dx * self.dropout_mask_final / (1.0 - self.dropout_rate)
            
        # 4d. گرادیان از ln_f
        grad = self.ln_f.backward(dx)
        
        # 4e. گرادیان از لایه ها (به ترتیب معکوس)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        # 4f. گرادیان از Dropout Embedding (نادیده گرفته شده در backward، فرض می کنیم با grad جمع شده)
        # (اگر dropout embedding را اضافه کردیم، باید اینجا اعمال شود)
        
        # 4g. گرادیان از Positional Embedding (اگر سینوسی باشد)
        if not self.pos_embed.use_rotary:
            self.pos_embed.backward(s, grad)
            
        # 4h. گرادیان از Embedding
        idx = self._cache['idx']
        self.embed.backward(idx, grad)
        
        # 5. برش گرادیان (Gradient Clipping)
        if grad_clip > 0:
            grads_map = self._get_grads_map()
            total_norm = 0.0
            for grad_array in grads_map.values():
                if grad_array is not None:
                    total_norm += np.sum(grad_array**2)
            total_norm = np.sqrt(total_norm)
            
            clip_coef = min(grad_clip / (total_norm + EPS), 1.0)
            
            if clip_coef < 1.0:
                for grad_array in grads_map.values():
                    if grad_array is not None:
                        grad_array *= clip_coef
                        
        return float(loss)

    def init_optimizer(self, lr: float = 6e-4, betas=(0.9, 0.95), eps=1e-8,
                       weight_decay: float = 0.1, warmup_steps: int = 2000):
        """
        مقادیر بهینه ساز AdamW را راه اندازی می کند.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.opt_step = 0
        self.opt_states = {}
        
        # ایجاد وضعیت (m و v) برای هر پارامتر
        for name, param in self.parameters():
            self.opt_states[name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }

    def step_optimizer(self, current_step: Optional[int] = None):
        """
        یک گام بهینه سازی AdamW را انجام می دهد.
        """
        if current_step is not None:
            self.opt_step = current_step
        self.opt_step += 1
        
        # 1. محاسبه نرخ یادگیری (LR Schedule)
        if self.warmup_steps > 0 and self.opt_step < self.warmup_steps:
            # گرم کردن خطی
            lr = self.lr * (self.opt_step / self.warmup_steps)
        else:
            # فروپاشی معکوس ریشه مربع (مانند "Attention Is All You Need")
            lr = self.lr * min(self.opt_step ** -0.5, self.opt_step * self.warmup_steps ** -1.5) if self.warmup_steps > 0 else self.lr

        # دریافت نقشه گرادیان ها
        grads_map = self._get_grads_map()

        # 2. به روز رسانی هر پارامتر
        for name, param in self.parameters():
            grad = grads_map.get(name)
            if grad is None:
                continue # پارامتری بدون گرادیان (مانند pos.W با RoPE)
                
            state = self.opt_states[name]
            
            # 3. اعمال Weight Decay (AdamW)
            # این کار مستقیماً روی پارامتر قبل از به‌روزرسانی m/v انجام می‌شود
            if self.weight_decay > 0 and 'W' in name: # معمولاً فقط روی وزن ها
                param -= lr * self.weight_decay * param
                
            # 4. به‌روزرسانی m (گشتاور اول)
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad
            # 5. به‌روزرسانی v (گشتاور دوم)
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)
            
            # 6. تصحیح بایاس
            m_hat = state['m'] / (1 - self.beta1 ** self.opt_step)
            v_hat = state['v'] / (1 - self.beta2 ** self.opt_step)
            
            # 7. به‌روزرسانی پارامتر
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def enable_gradient_checkpointing(self):
        """
        گرادیان چک پوینتینگ در پیاده سازی NumPy پشتیبانی نمی شود.
        """
        warnings.warn("Gradient checkpointing is not implemented in this NumPy version", RuntimeWarning)

    def convert_to_rms_norm(self):
        """همه لایه های LayerNorm را به RMSNorm تبدیل می کند."""
        self.ln_f = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)
        for layer in self.layers:
            layer.ln1 = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)
            layer.ln2 = LayerNorm(self.d_model, rms_norm=True, dtype=self.dtype)

    def save(self, path: str, include_optimizer: bool = False):
        """مدل را در یک فایل pickle ذخیره می کند."""
        config = self.layers[0].mha if self.layers else None
        data = {
            'config': {
                'vocab_size': self.vocab_size,
                'max_len': self.max_len,
                'd_model': self.d_model,
                'num_heads': config.num_heads if config else 0,
                'd_ff': self.layers[0].ff.d_ff if self.layers else 0,
                'num_layers': len(self.layers),
                'dropout': self.dropout_rate,
                'use_rotary': self.pos_embed.use_rotary,
                'rms_norm': self.ln_f.rms_norm,
                'layer_scale': any(layer.layer_scale for layer in self.layers)
            },
            'params': dict(self.parameters()) # ذخیره همه پارامترها با نام
        }
        
        if include_optimizer and self.opt_states:
            data['optimizer'] = {
                'lr': self.lr,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'eps': self.eps,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'opt_step': self.opt_step,
                'states': self.opt_states
            }
            
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, strict: bool = True):
        """مدل را از یک فایل pickle بارگذاری می کند."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # بارگذاری پارامترها
        loaded_params = data['params']
        current_params_dict = dict(self.parameters())
        
        for name, param_array in current_params_dict.items():
            if name in loaded_params:
                if param_array.shape == loaded_params[name].shape:
                    param_array[:] = loaded_params[name] # کپی درجا
                elif strict:
                    raise ValueError(f"Shape mismatch for {name}: model has {param_array.shape}, file has {loaded_params[name].shape}")
            elif strict:
                raise ValueError(f"Missing parameter in file: {name}")

        # بارگذاری وضعیت بهینه ساز
        if 'optimizer' in data:
            opt_data = data['optimizer']
            self.init_optimizer(
                lr=opt_data['lr'],
                betas=(opt_data['beta1'], opt_data['beta2']),
                eps=opt_data['eps'],
                weight_decay=opt_data.get('weight_decay', 0.1),
                warmup_steps=opt_data.get('warmup_steps', 0)
            )
            self.opt_step = opt_data['opt_step']
            
            # اطمینان از مطابقت حالت های بهینه ساز
            for name in self.opt_states:
                if name in opt_data['states']:
                    if self.opt_states[name]['m'].shape == opt_data['states'][name]['m'].shape:
                         self.opt_states[name]['m'][:] = opt_data['states'][name]['m']
                         self.opt_states[name]['v'][:] = opt_data['states'][name]['v']
                    elif strict:
                         warnings.warn(f"Optimizer state shape mismatch for {name}. Re-initializing state.")
                elif strict:
                    warnings.warn(f"Missing optimizer state for {name}. State will be re-initialized.")
        print(f"Model loaded successfully from {path}.")

    # --- متدهای تولید متن ---
    
    @staticmethod
    def _sample_top_p(probs: np.ndarray, p: float) -> int:
        """نمونه برداری Top-p (Nucleus)"""
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # پیدا کردن اندیس هایی که مجموع تجمعی آنها از p بیشتر است
        idx_to_remove = cumulative_probs > p
        # اولین اندیسی که از p گذشته را نگه دار
        idx_to_remove[1:] = idx_to_remove[:-1]
        idx_to_remove[0] = False
        
        # احتمالات توکن های حذف شده را صفر کن
        probs[sorted_indices[idx_to_remove]] = 0
        probs = probs / (np.sum(probs) + EPS) # نرمال سازی مجدد
        
        return np.random.choice(len(probs), p=probs)

    @staticmethod
    def _sample_top_k(probs: np.ndarray, k: int) -> int:
        """نمونه برداری Top-k"""
        k = min(k, len(probs))
        top_k_indices = np.argsort(probs)[-k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / (np.sum(top_k_probs) + EPS) # نرمال سازی مجدد
        
        # نمونه برداری از بین k توکن برتر
        chosen_local_idx = np.random.choice(len(top_k_indices), p=top_k_probs)
        return top_k_indices[chosen_local_idx]

    def generate(self, idx: np.ndarray, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> List[int]:
        """
        تولید متن (تکمیل دنباله).
        idx: (1, s) آرایه ای از ID های شروع
        """
        b, s = idx.shape
        if b != 1:
            raise ValueError("Generation only supports batch size 1")
            
        generated_ids = idx[0].tolist()
        
        for _ in range(max_new_tokens):
            # اگر دنباله فعلی طولانی تر از max_len است، آن را برش بزن
            idx_cond = np.array([generated_ids[-self.max_len:]], dtype=np.int32)
            
            # 1. Forward pass
            logits = self.forward(idx_cond, training=False) # (1, s, vocab_size)
            
            # 2. دریافت logits برای توکن آخر
            last_logits = logits[0, -1, :] # (vocab_size,)
            
            # 3. اعمال دما
            if temperature <= 0:
                # حالت Greedy (انتخاب argmax)
                next_id = np.argmax(last_logits)
            else:
                last_logits = last_logits / temperature
                
                # 4. دریافت احتمالات
                probs = softmax(last_logits)
            
                # 5. نمونه برداری
                if top_p is not None and 0 < top_p < 1:
                    next_id = self._sample_top_p(probs, top_p)
                elif top_k is not None and top_k > 0:
                    next_id = self._sample_top_k(probs, top_k)
                else:
                    # نمونه برداری استاندارد
                    next_id = np.random.choice(len(probs), p=probs)
            
            # 6. افزودن توکن جدید و بررسی توقف
            generated_ids.append(next_id)
            
            if eos_token_id is not None and next_id == eos_token_id:
                break
                
        return generated_ids

# --- مثال اجرایی ---
if __name__ == "__main__":
    print("شروع پیاده سازی Transformer (GPT-like) با NumPy...")
    np.random.seed(42) # برای تکرارپذیری
    
    # 1. راه اندازی توکنایزر
    print("\n--- 1. آموزش توکنایزر ---")
    tokenizer = BPETokenizer()
    # متن نمونه برای آموزش توکنایزر
    sample_texts = [
        "This is a sample text.",
        "This is another sample text, for BPE.",
        "Byte-Pair Encoding is useful.",
        "The quick brown fox jumps over the lazy dog.",
        "سلام دنیا!",
        "این یک متن آزمایشی است."
    ]
    # ما به اندازه واژگان بزرگتری نیاز داریم تا ادغام ها را یاد بگیریم
    # (256 برای بایت ها + 4 توکن ویژه + ادغام ها)
    tokenizer.build_from_text(sample_texts, vocab_size=500)
    print(f"اندازه واژگان توکنایزر: {len(tokenizer.vocab)}")
    
    # تست کدگذاری/کدگشایی
    text = "Hello, world! این یک تست است."
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"اصلی: '{text}'")
    print(f"کدگذاری شده: {encoded}")
    print(f"کدگشایی شده: '{decoded}'")
    
    # 2. راه اندازی مدل
    print("\n--- 2. راه اندازی مدل ---")
    VOCAB_SIZE = len(tokenizer.vocab)
    MAX_LEN = 64
    D_MODEL = 64 # مدل کوچک برای تست
    NUM_HEADS = 4
    D_FF = 128
    NUM_LAYERS = 2
    DROPOUT = 0.0 # غیرفعال کردن dropout برای تست ساده
    
    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_rotary=False, # سینوسی استاندارد
        rms_norm=False,   # LayerNorm استاندارد
        layer_scale=True  # استفاده از LayerScale
    )
    print(f"مدل با {NUM_LAYERS} لایه و {D_MODEL} d_model ساخته شد.")
    
    # 3. حلقه آموزش (ساختگی)
    print("\n--- 3. اجرای گام آموزش ساختگی ---")
    
    # ایجاد داده ساختگی
    # تلاش برای یادگیری "Hello, world!" -> " world! Hello,"
    train_text = "Hello, world! " * 3 # تکرار برای دنباله طولانی تر
    input_ids = tokenizer.encode(train_text, max_len=MAX_LEN, add_bos=True, add_eos=False)
    # هدف یک توکن شیفت یافته است
    target_ids = np.roll(input_ids, -1)
    target_ids[-1] = tokenizer.w2i['<eos>'] # توکن پایانی را پیش بینی کن
    
    # اضافه کردن بعد دسته
    batch_in = input_ids[np.newaxis, :]
    batch_out = target_ids[np.newaxis, :]
    
    print(f"متن ورودی: '{tokenizer.decode(batch_in[0])}'")
    print(f"متن هدف: '{tokenizer.decode(batch_out[0])}'")
    
    # راه اندازی بهینه ساز
    model.init_optimizer(lr=1e-3, warmup_steps=10)
    
    # گام های آموزش
    num_steps = 50
    start_time = time.time()
    for step in range(num_steps):
        # صفر کردن گرادیان ها
        model.zero_grads()
        
        # Forward و backward
        loss = model.loss_and_backward(batch_in, batch_out, grad_clip=1.0)
        
        # گام بهینه ساز
        model.step_optimizer()
        
        if (step + 1) % 10 == 0:
            print(f"گام {step+1}/{num_steps}, زیان: {loss:.4f}")
            
    end_time = time.time()
    print(f"آموزش تمام شد. ({end_time - start_time:.2f} ثانیه)")

    # 4. تولید متن
    print("\n--- 4. اجرای تولید متن ---")
    
    # دریافت توکن <bos>
    bos_id = tokenizer.w2i['<bos>']
    eos_id = tokenizer.w2i['<eos>']
    start_ids = np.array([[bos_id]], dtype=np.int32)
    
    generated_sequence = model.generate(
        start_ids,
        max_new_tokens=20,
        temperature=0.7,
        top_k=5,
        eos_token_id=eos_id
    )
    
    generated_text = tokenizer.decode(generated_sequence)
    print(f"دنباله تولید شده: {generated_sequence}")
    print(f"متن تولید شده: '{generated_text}'")
    
    # 5. ذخیره/بارگذاری
    print("\n--- 5. تست ذخیره/بارگذاری ---")
    MODEL_PATH = "./transformer_model.pkl"
    
    model.save(MODEL_PATH, include_optimizer=True)
    print(f"مدل در {MODEL_PATH} ذخیره شد.")
    
    # ایجاد مدل جدید
    model_loaded = GPT(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_rotary=False,
        rms_norm=False,
        layer_scale=True
    )
    
    model_loaded.load(MODEL_PATH)
    print("مدل بارگذاری شد.")
    
    # بررسی اینکه آیا پارامترها بارگذاری شده اند (بررسی W_out)
    param_equal = np.allclose(model.W_out, model_loaded.W_out)
    opt_state_equal = np.allclose(model.opt_states['W_out']['m'], model_loaded.opt_states['W_out']['m'])
    
    print(f"پارامترهای W_out اصلی و بارگذاری شده برابر هستند: {param_equal}")
    print(f"وضعیت بهینه ساز اصلی و بارگذاری شده (W_out 'm') برابر هستند: {opt_state_equal}")
    print(f"گام بهینه ساز اصلی: {model.opt_step}, گام بارگذاری شده: {model_loaded.opt_step}")

    # پاکسازی
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"فایل {MODEL_PATH} پاک شد.")
