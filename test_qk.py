import math
from typing import Optional
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel



def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    # if enable_gqa:
    #     key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    #     value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

q = torch.randn(1, 2, 7, 101, 64)
k = torch.randn(1, 2, 1, 101, 64)
v = torch.randn(1, 2, 1, 101, 64)
mask = torch.zeros(1, 2, 7, 101, 101)

y1 = scaled_dot_product_attention(
    q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0, is_causal=mask is None)

q2 = q.reshape(1, -1, 101, 64)
k2 = torch.repeat_interleave(k, 7, 2).reshape(1, -1, 101, 64)
v2 = torch.repeat_interleave(v, 7, 2).reshape(1, -1, 101, 64)
mask2 = mask.reshape(1, -1, 101, 101)
y2 = scaled_dot_product_attention(
    q2, k2, v2, attn_mask=mask2, dropout_p=0.0, scale=1.0, is_causal=mask2 is None)
 
assert torch.allclose(y1, y2.reshape(1, 2, 7, 101, 64))

with sdpa_kernel([SDPBackend.MATH]):
  y3 = torch.nn.functional.scaled_dot_product_attention(
      q2, k2, v2, attn_mask=mask2, dropout_p=0.0, scale=1.0, is_causal=mask2 is None)
 
assert torch.allclose(y2, y3)

# y3_1 = torch.nn.functional.scaled_dot_product_attention(
#     q2, k2, v2, attn_mask=mask2, dropout_p=0.0, scale=1.0, is_causal=mask2 is None)
 
# assert torch.allclose(y3, y3_1)


q3 = q
k3 = torch.repeat_interleave(k, 7, 2)
v3 = torch.repeat_interleave(v, 7, 2)
mask3 = mask
with sdpa_kernel([SDPBackend.MATH]):
  y4 = torch.nn.functional.scaled_dot_product_attention(
      q3, k3, v3, attn_mask=mask3, dropout_p=0.0, scale=1.0, is_causal=mask3 is None)

assert torch.allclose(y3, y4.reshape(1, -1, 101, 64))

q4 = q
k4 = k
v4 = v
mask4 = mask
with sdpa_kernel([SDPBackend.MATH]):
  y5 = torch.nn.functional.scaled_dot_product_attention(
      q4, k4, v4, attn_mask=mask4, dropout_p=0.0, scale=1.0, is_causal=mask3 is None)

assert torch.allclose(y4, y5)
# q = torch.randn(1, 2, 7, 101, 64)
# k = torch.randn(1, 2, 1, 101, 64)

# a = q @ k.transpose(3,4)
# b = q @ torch.repeat_interleave(k, 7, 2).transpose(3,4)
# assert torch.allclose(a, b)

