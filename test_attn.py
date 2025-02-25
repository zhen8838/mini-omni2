import torch

x: torch.Tensor[seq_len, embed_size] # [101, 768]
qkv: torch.Tensor[seq_len, (q_groups * qkv head_nums) * head_size] = attn_liner(x) # [101, 1152]
qkv: torch.Tensor[seq_len, q_groups, qkv head_nums, head_size] = reshape(qkv, (seq_len, q_groups, qkv head_nums, head_size)) # [101, 2, 9, 64]
qkv: torch.Tensor[q_groups, qkv head_nums, seq_len, head_size] = permute(qkv, [1,2,0,3]) # [2, 9, 101, 64]
(q, k, v) = split(qkv, (q_per_kv, 1, 1), dim=1)

q: torch.Tensor[q_groups, q_nums, seq_len, head_size] # [2, 7, 101, 64]
k: torch.Tensor[q_groups, k_nums, seq_len, head_size] # [2, 1, 101, 64]
v: torch.Tensor[q_groups, v_nums, seq_len, head_size] # [2, 1, 101, 64]

q: torch.Tensor[q_groups, q_nums, seq_len, head_size] = rope(q) # [2, 7, 101, 64]
k: torch.Tensor[q_groups, k_nums, seq_len, head_size] = rope(k) # [2, 1, 101, 64]

k, v = kv_cache(past_k, past_v, k, v)
k: torch.Tensor[q_groups, k_nums, seq_len + hist_len, head_size] # [2, 1, 101, 64]
v: torch.Tensor[q_groups, v_nums, seq_len + hist_len, head_size] # [2, 1, 101, 64]

y: torch.Tensor[q_groups, q_nums, seq_len + hist_len, head_size] = scaled_dot_product_attention(q,k,v) # [2, 7, 101, 64]

y: torch.Tensor[seq_len + hist_len, q_groups, q_nums, head_size] = torch.permute(y, [2, 0, 1, 3]) # [101, 2, 7, 64]

y: torch.Tensor[seq_len + hist_len, q_groups * q_nums * head_size] = torch.reshape(y) # [101, 896]
