from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.models import model as openpi_model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from rlinf.models.embodiment.openpi import get_model

ROOT = Path('/data/cyy/RLinf/tests/pi05_snapshot')
CHECKPOINT = Path('/data/cyy/ckpts/adk111/piper_pi05_SFT_right_place_cup_on_book_left_insert_30000')


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        tensor = x.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    return np.asarray(x)


def pack(x: Any, n: int = 8) -> dict[str, Any]:
    a = to_numpy(x)
    if a.dtype == np.bool_:
        a_hash = hashlib.md5(a.astype(np.uint8).tobytes()).hexdigest()
    else:
        a_hash = hashlib.md5(a.tobytes()).hexdigest()
    flat = a.reshape(-1) if a.size else a
    return {
        'shape': list(a.shape),
        'dtype': str(a.dtype),
        'first': flat[:n].tolist() if a.size else [],
        'sum': float(a.astype(np.float64).sum()) if a.size else 0.0,
        'mean': float(a.astype(np.float64).mean()) if a.size else 0.0,
        'std': float(a.astype(np.float64).std()) if a.size else 0.0,
        'hash': a_hash,
    }


def hash_tree(obj: Any) -> str:
    h = hashlib.md5()
    def visit(x: Any):
        if isinstance(x, torch.Tensor):
            tensor = x.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            arr = tensor.numpy()
            h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
        elif isinstance(x, (list, tuple)):
            h.update(f'{type(x).__name__}:{len(x)}'.encode())
            for item in x:
                visit(item)
        elif isinstance(x, dict):
            h.update(f'dict:{len(x)}'.encode())
            for key in sorted(x):
                h.update(str(key).encode())
                visit(x[key])
        else:
            h.update(repr(x).encode())
    visit(obj)
    return h.hexdigest()


def load_snapshot():
    data = np.load(ROOT / 'sample_frame.npz', allow_pickle=True)
    return {
        'left_wrist': data['left_wrist'],
        'right_wrist': data['right_wrist'],
        'right_high': data['right_high'],
        'state': data['state'].astype(np.float32),
        'task': str(data['task'].tolist()),
        'frame_index': int(data['frame_index']),
    }


def main():
    snapshot = load_snapshot()
    noise = torch.from_numpy(np.load(ROOT / 'noise.npy')).to(torch.float32)

    cfg = OmegaConf.create({
        'model_path': str(CHECKPOINT),
        'openpi': {
            'config_name': 'pi05_piper',
            'lerobot_compat': True,
        },
        'openpi_data': {
            'env_action_dim': 14,
            'align_evorl_observation': True,
        },
    })
    model = get_model(cfg)
    model.eval()

    env_obs = {
        'main_images': torch.from_numpy(snapshot['right_high'][None, ...]),
        'wrist_images': torch.from_numpy(snapshot['left_wrist'][None, ...]),
        'extra_view_images': torch.from_numpy(np.stack([snapshot['left_wrist'], snapshot['right_wrist']], axis=0)[None, ...]),
        'states': torch.from_numpy(snapshot['state'][None, ...]),
        'task_descriptions': [snapshot['task']],
    }

    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = openpi_model.Observation.from_dict(processed_obs)

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation_lerobot_compat(observation)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    (_, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    num_steps = model.config.num_steps
    dt = -1.0 / num_steps
    x_t = noise.clone().to(model.action_in_proj.weight.dtype)
    trace_steps = []
    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.full((x_t.shape[0],), time, dtype=torch.float32, device=x_t.device)
        suffix_out = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, timestep)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        v_t = model.action_out_proj(suffix_out)
        x_next = x_t + dt * v_t
        trace_steps.append({
            'step': step,
            'time': time,
            'suffix_embs': pack(suffix_embs),
            'suffix_pad_masks': pack(suffix_pad_masks.to(torch.int32)),
            'suffix_att_masks': pack(suffix_att_masks.to(torch.int32)),
            'adarms_cond': pack(adarms_cond),
            'full_att_2d_masks': pack(full_att_2d_masks.to(torch.int32)),
            'position_ids': pack(position_ids),
            'suffix_out': pack(suffix_out),
            'v_t': pack(v_t),
            'x_t_in': pack(x_t),
            'x_t_out': pack(x_next),
        })
        x_t = x_next

    builtin = model.sample_actions(observation, noise=noise.clone(), mode='eval', compute_values=False)['actions']

    result = {
        'snapshot': {
            'frame_index': snapshot['frame_index'],
            'task': snapshot['task'],
            'state': pack(snapshot['state']),
            'left_wrist': pack(snapshot['left_wrist']),
            'right_wrist': pack(snapshot['right_wrist']),
            'right_high': pack(snapshot['right_high']),
            'noise': pack(noise),
        },
        'obs_processor': {
            'observation/image': pack(to_process_obs['observation/image']),
            'observation/wrist_image': pack(to_process_obs['observation/wrist_image']),
            'observation/extra_view_image': pack(to_process_obs['observation/extra_view_image']),
            'observation/state': pack(to_process_obs['observation/state']),
            'prompt': to_process_obs['prompt'][0] if isinstance(to_process_obs['prompt'], list) else to_process_obs['prompt'],
        },
        'input_transform': {
            'image.base_0_rgb': pack(processed_obs['image']['base_0_rgb']),
            'image.left_wrist_0_rgb': pack(processed_obs['image']['left_wrist_0_rgb']),
            'image.right_wrist_0_rgb': pack(processed_obs['image']['right_wrist_0_rgb']),
            'image_mask.base_0_rgb': pack(processed_obs['image_mask']['base_0_rgb'].to(torch.int32)),
            'image_mask.left_wrist_0_rgb': pack(processed_obs['image_mask']['left_wrist_0_rgb'].to(torch.int32)),
            'image_mask.right_wrist_0_rgb': pack(processed_obs['image_mask']['right_wrist_0_rgb'].to(torch.int32)),
            'state': pack(processed_obs['state']),
            'tokenized_prompt': pack(processed_obs['tokenized_prompt']),
            'tokenized_prompt_mask': pack(processed_obs['tokenized_prompt_mask'].to(torch.int32)),
        },
        'model_inputs': {
            'images.base_0_rgb': pack(images[0]),
            'images.left_wrist_0_rgb': pack(images[1]),
            'images.right_wrist_0_rgb': pack(images[2]),
            'img_masks.base_0_rgb': pack(img_masks[0].to(torch.int32)),
            'img_masks.left_wrist_0_rgb': pack(img_masks[1].to(torch.int32)),
            'img_masks.right_wrist_0_rgb': pack(img_masks[2].to(torch.int32)),
            'lang_tokens': pack(lang_tokens),
            'lang_masks': pack(lang_masks.to(torch.int32)),
            'state': pack(state),
        },
        'prefix': {
            'prefix_embs': pack(prefix_embs),
            'prefix_pad_masks': pack(prefix_pad_masks.to(torch.int32)),
            'prefix_att_masks': pack(prefix_att_masks.to(torch.int32)),
            'prefix_att_2d_masks': pack(prefix_att_2d_masks.to(torch.int32)),
            'prefix_position_ids': pack(prefix_position_ids),
            'prefix_att_2d_masks_4d': pack(prefix_att_2d_masks_4d),
            'past_key_values_hash': hash_tree(past_key_values),
        },
        'denoise_steps': trace_steps,
        'final': {
            'manual_actions': pack(x_t),
            'builtin_actions': pack(builtin),
            'manual_builtin_equal': bool(torch.equal(x_t, builtin)),
            'manual_builtin_max_abs_diff': float((x_t - builtin).abs().max()),
        },
    }
    (ROOT / 'rlinf_trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        'saved': str(ROOT / 'rlinf_trace.json'),
        'manual_builtin_equal': result['final']['manual_builtin_equal'],
        'manual_builtin_max_abs_diff': result['final']['manual_builtin_max_abs_diff'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
