# patch_attention.py
# Replaces flash_attn with PyTorch native SDPA
# Run automatically during installation

import os

attention_file = "wan/modules/attention.py"

if not os.path.exists(attention_file):
    print(f"File not found: {attention_file}")
    exit(1)

with open(attention_file, "r", encoding="utf-8") as f:
    content = f.read()

# Backup original
with open(attention_file + ".original", "w", encoding="utf-8") as f:
    f.write(content)

# New implementation using PyTorch SDPA (no flash_attn needed)
new_flash_attention = '''def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, causal=False, window_size=(-1, -1), **kwargs):
    import torch.nn.functional as F
    
    # Transpose from (B, S, H, D) to (B, H, S, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
    
    # Transpose back to (B, S, H, D)
    return out.transpose(1, 2)
'''

# Find and replace the flash_attention function
if "def flash_attention(" in content:
    # Find the start of the function
    start_idx = content.find("def flash_attention(")
    
    # Find the end (next function/class at same indentation level)
    rest = content[start_idx:]
    lines = rest.split('\n')
    end_line_idx = len(lines)
    
    for i, line in enumerate(lines[1:], 1):
        # New top-level definition
        if line and not line.startswith(' ') and not line.startswith('\t'):
            if line.startswith('def ') or line.startswith('class ') or line.startswith('@'):
                end_line_idx = i
                break
    
    # Reconstruct
    before = content[:start_idx]
    after_lines = lines[end_line_idx:]
    
    new_content = before + new_flash_attention + '\n'.join(after_lines)
    
    with open(attention_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("Patched wan/modules/attention.py - flash_attn replaced with PyTorch SDPA")
else:
    print("flash_attention function not found - skipping patch")
