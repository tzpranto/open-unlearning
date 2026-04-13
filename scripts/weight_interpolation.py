"""
Weight interpolation between T8f (fk=0.428, rk=0.461, ABOVE +0.056) 
and U4 (fk=0.328, rk=0.330, ON frontier).

If the loss landscape between these two checkpoints is smooth,
linear interpolation should give intermediate points that may be ABOVE the frontier.
"""
import sys
import os
import json
import shutil
from safetensors.torch import load_file, save_file

BASE = "/datadrive/forked/open-unlearning/saves/unlearn"
T8F_DIR = os.path.join(BASE, "ablation_T8f_weighted_contrastive")
U4_DIR = os.path.join(BASE, "ablation_U4_k5")

def interpolate(alpha, out_name):
    """Create interpolated model: alpha*T8f + (1-alpha)*U4"""
    out_dir = os.path.join(BASE, out_name)
    if os.path.exists(out_dir) and any(f.endswith('.safetensors') for f in os.listdir(out_dir) if 'model-' in f):
        print(f"[SKIP] {out_name} already exists")
        return out_dir
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Copy config files from T8f (same architecture)
    for f in ['config.json', 'generation_config.json', 'special_tokens_map.json', 
              'tokenizer.json', 'tokenizer_config.json']:
        src = os.path.join(T8F_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
    
    # Interpolate each shard
    for shard in ['model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors']:
        t8f_path = os.path.join(T8F_DIR, shard)
        u4_path = os.path.join(U4_DIR, shard)
        
        print(f"  Loading {shard}...")
        t8f_tensors = load_file(t8f_path)
        u4_tensors = load_file(u4_path)
        
        blended = {}
        for key in t8f_tensors:
            blended[key] = alpha * t8f_tensors[key] + (1 - alpha) * u4_tensors[key]
        
        out_path = os.path.join(out_dir, shard)
        save_file(blended, out_path)
        print(f"  Saved {shard}")
        
        del t8f_tensors, u4_tensors, blended
    
    # Copy index file
    idx_src = os.path.join(T8F_DIR, 'model.safetensors.index.json')
    if os.path.exists(idx_src):
        shutil.copy2(idx_src, out_dir)
    
    print(f"[DONE] {out_name} (alpha={alpha})")
    return out_dir

if __name__ == "__main__":
    alphas = [0.7, 0.5, 0.3]  # More T8f, equal, more U4
    for a in alphas:
        name = f"ablation_V_interp_a{int(a*10)}"
        print(f"\n=== Interpolating alpha={a} ({name}) ===")
        interpolate(a, name)
