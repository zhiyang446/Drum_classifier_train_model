# -*- coding: utf-8 -*-
import torch

def main():
    device = torch.device('cpu')

    state_orig = torch.load('mixed_formal_kick375_snare18_hh12_candidate.pth', map_location=device, weights_only=False)
    state_v13 = torch.load('validation_runs/six_class_candidate_v13/six_class_candidate_v13.pth', map_location=device, weights_only=False)

    # 統一 legacy 投影名以利比對
    if 'backbone.slot_proj.weight' in state_orig:
        state_orig = dict(state_orig)
        state_orig['backbone.legacy_slot_proj.weight'] = state_orig.pop('backbone.slot_proj.weight')
        state_orig['backbone.legacy_slot_proj.bias'] = state_orig.pop('backbone.slot_proj.bias')

    if 'backbone.slot_proj.weight' in state_v13:
        state_v13 = dict(state_v13)
        state_v13['backbone.legacy_slot_proj.weight'] = state_v13.pop('backbone.slot_proj.weight')
        state_v13['backbone.legacy_slot_proj.bias'] = state_v13.pop('backbone.slot_proj.bias')

    print("Comparing ALL parameters:")
    different_keys = []
    for key in state_orig.keys():
        if key in state_v13:
            orig_tensor = state_orig[key]
            v13_tensor = state_v13[key]
            if key.startswith('onset_head') or key.startswith('velocity_head'):
                diff = (orig_tensor - v13_tensor[:3]).abs().max().item()
            else:
                if orig_tensor.shape == v13_tensor.shape:
                    diff = (orig_tensor - v13_tensor).abs().max().item()
                else:
                    print(f"Shape mismatch for {key}: orig {orig_tensor.shape}, v13 {v13_tensor.shape}")
                    different_keys.append((key, "shape mismatch"))
                    continue
            if diff > 1e-6:
                print(f"Parameter {key} changed! diff = {diff}")
                different_keys.append((key, diff))
        else:
            print(f"Key {key} not found in state_v13")
            different_keys.append((key, "missing"))

    if not different_keys:
        print("All matching parameter values are 100% IDENTICAL!")
    else:
        print(f"Found {len(different_keys)} differences/mismatches.")

if __name__ == '__main__':
    main()
