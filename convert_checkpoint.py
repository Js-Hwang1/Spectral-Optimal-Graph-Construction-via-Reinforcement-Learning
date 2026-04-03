#!/usr/bin/env python3
"""
Convert between C binary checkpoints (.bin) and PyTorch checkpoints (.pt).

Usage:
    python convert_checkpoint.py bin2pt input.bin output.pt
    python convert_checkpoint.py pt2bin input.pt output.bin
"""

import struct
import sys
import numpy as np

HIDDEN_DIM = 64
EDGE_FEAT_DIM = 4
GRAPH_FEAT_DIM = 3

# Header: magic(4) + version(4) + hidden_dim(4) + edge_feat_dim(4) +
#          graph_feat_dim(4) + episode(4) + best_avg_improvement(4) + reserved(20)
HEADER_FMT = '<IIIIIIfIIIII'  # 48 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRL_CKPT_MAGIC = 0x43524C42


def read_mlp_weights(f, in_dim):
    """Read one MLP's weights from binary file."""
    W0 = np.frombuffer(f.read(HIDDEN_DIM * in_dim * 4), dtype=np.float32).reshape(HIDDEN_DIM, in_dim)
    b0 = np.frombuffer(f.read(HIDDEN_DIM * 4), dtype=np.float32).copy()
    W1 = np.frombuffer(f.read(HIDDEN_DIM * HIDDEN_DIM * 4), dtype=np.float32).reshape(HIDDEN_DIM, HIDDEN_DIM)
    b1 = np.frombuffer(f.read(HIDDEN_DIM * 4), dtype=np.float32).copy()
    W2 = np.frombuffer(f.read(HIDDEN_DIM * 4), dtype=np.float32).reshape(1, HIDDEN_DIM)
    b2 = np.frombuffer(f.read(1 * 4), dtype=np.float32).copy()
    return W0.copy(), b0, W1.copy(), b1, W2.copy(), b2


def write_mlp_weights(f, W0, b0, W1, b1, W2, b2):
    """Write one MLP's weights to binary file."""
    f.write(W0.astype(np.float32).tobytes())
    f.write(b0.astype(np.float32).tobytes())
    f.write(W1.astype(np.float32).tobytes())
    f.write(b1.astype(np.float32).tobytes())
    f.write(W2.astype(np.float32).tobytes())
    f.write(b2.astype(np.float32).tobytes())


def bin2pt(bin_path, pt_path):
    """Convert .bin checkpoint to .pt format compatible with eval_refine.py."""
    import torch

    with open(bin_path, 'rb') as f:
        header_data = f.read(HEADER_SIZE)
        fields = struct.unpack(HEADER_FMT, header_data)
        magic = fields[0]
        assert magic == CRL_CKPT_MAGIC, f"Bad magic: 0x{magic:08X}"
        episode = fields[5]
        best_imp = fields[6]

        add_W0, add_b0, add_W1, add_b1, add_W2, add_b2 = read_mlp_weights(f, EDGE_FEAT_DIM)
        rem_W0, rem_b0, rem_W1, rem_b1, rem_W2, rem_b2 = read_mlp_weights(f, EDGE_FEAT_DIM)
        val_W0, val_b0, val_W1, val_b1, val_W2, val_b2 = read_mlp_weights(f, GRAPH_FEAT_DIM)

    state_dict = {
        'add_mlp.0.weight': torch.from_numpy(add_W0),
        'add_mlp.0.bias': torch.from_numpy(add_b0),
        'add_mlp.2.weight': torch.from_numpy(add_W1),
        'add_mlp.2.bias': torch.from_numpy(add_b1),
        'add_mlp.4.weight': torch.from_numpy(add_W2),
        'add_mlp.4.bias': torch.from_numpy(add_b2),
        'rem_mlp.0.weight': torch.from_numpy(rem_W0),
        'rem_mlp.0.bias': torch.from_numpy(rem_b0),
        'rem_mlp.2.weight': torch.from_numpy(rem_W1),
        'rem_mlp.2.bias': torch.from_numpy(rem_b1),
        'rem_mlp.4.weight': torch.from_numpy(rem_W2),
        'rem_mlp.4.bias': torch.from_numpy(rem_b2),
        'value_mlp.0.weight': torch.from_numpy(val_W0),
        'value_mlp.0.bias': torch.from_numpy(val_b0),
        'value_mlp.2.weight': torch.from_numpy(val_W1),
        'value_mlp.2.bias': torch.from_numpy(val_b1),
        'value_mlp.4.weight': torch.from_numpy(val_W2),
        'value_mlp.4.bias': torch.from_numpy(val_b2),
    }

    torch.save({
        'policy_state_dict': state_dict,
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'edge_feat_dim': EDGE_FEAT_DIM,
            'graph_feat_dim': GRAPH_FEAT_DIM,
        },
        'episode': episode,
        'metrics': {
            'best_avg_improvement': best_imp,
        },
    }, pt_path)

    print(f"Converted {bin_path} -> {pt_path} (episode={episode})")


def pt2bin(pt_path, bin_path):
    """Convert .pt checkpoint to .bin format for C training."""
    import torch

    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['policy_state_dict']
    episode = ckpt.get('episode', 0)
    metrics = ckpt.get('metrics', {})
    best_imp = metrics.get('best_avg_improvement', 0.0)

    with open(bin_path, 'wb') as f:
        header = struct.pack(HEADER_FMT,
                             CRL_CKPT_MAGIC, 1, HIDDEN_DIM, EDGE_FEAT_DIM,
                             GRAPH_FEAT_DIM, episode,
                             struct.unpack('I', struct.pack('f', best_imp))[0],
                             0, 0, 0, 0, 0)
        f.write(header)

        def write_mlp(prefix):
            W0 = sd[f'{prefix}.0.weight'].numpy()
            b0 = sd[f'{prefix}.0.bias'].numpy()
            W1 = sd[f'{prefix}.2.weight'].numpy()
            b1 = sd[f'{prefix}.2.bias'].numpy()
            W2 = sd[f'{prefix}.4.weight'].numpy()
            b2 = sd[f'{prefix}.4.bias'].numpy()
            write_mlp_weights(f, W0, b0, W1, b1, W2, b2)

        write_mlp('add_mlp')
        write_mlp('rem_mlp')
        write_mlp('value_mlp')

    print(f"Converted {pt_path} -> {bin_path} (episode={episode})")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python convert_checkpoint.py <bin2pt|pt2bin> <input> <output>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'bin2pt':
        bin2pt(sys.argv[2], sys.argv[3])
    elif mode == 'pt2bin':
        pt2bin(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown mode: {mode}. Use 'bin2pt' or 'pt2bin'.")
        sys.exit(1)
