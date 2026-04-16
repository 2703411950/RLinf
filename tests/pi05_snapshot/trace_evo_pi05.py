from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

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
            arr = x.detach().cpu().numpy()
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

    raw_obs = {
        'observation.images.left_wrist': snapshot['left_wrist'],
        'observation.images.right_wrist': snapshot['right_wrist'],
        'observation.images.right_high': snapshot['right_high'],
        'observation.state': snapshot['state'],
    }
    prepared = prepare_observation_for_inference(
        dict(raw_obs),
        device=torch.device('cpu'),
        task=snapshot['task'],
        robot_type='bi_piper_follower',
    )

    policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT))
    policy_cfg.device = 'cpu'
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )

    transition = batch_to_transition(dict(prepared))
    preprocessor_steps = []
    for step in preprocessor.steps:
        transition = step(transition)
        batch = transition_to_batch(transition)
        entry = {'step': type(step).__name__}
        if OBS_STATE in batch:
            entry['state'] = pack(batch[OBS_STATE])
        if 'task' in batch:
            entry['task'] = batch['task'][0] if isinstance(batch['task'], list) else batch['task']
        if OBS_LANGUAGE_TOKENS in batch:
            entry['tokenized_prompt'] = pack(batch[OBS_LANGUAGE_TOKENS])
        if OBS_LANGUAGE_ATTENTION_MASK in batch:
            entry['tokenized_prompt_mask'] = pack(batch[OBS_LANGUAGE_ATTENTION_MASK].to(torch.int32))
        for key in ['observation.images.right_high', 'observation.images.left_wrist', 'observation.images.right_wrist']:
            if key in batch:
                entry[key] = pack(batch[key])
        for alias_key, src_key in {
            'observation.images.cam_high': 'observation.images.right_high',
            'observation.images.cam_left_wrist': 'observation.images.left_wrist',
            'observation.images.cam_right_wrist': 'observation.images.right_wrist',
        }.items():
            if src_key in batch:
                batch[alias_key] = batch[src_key]
        preprocessor_steps.append(entry)

    batch = transition_to_batch(transition)
    batch['observation.images.cam_high'] = batch['observation.images.right_high']
    batch['observation.images.cam_left_wrist'] = batch['observation.images.left_wrist']
    batch['observation.images.cam_right_wrist'] = batch['observation.images.right_wrist']
    policy = PI05Policy.from_pretrained(str(CHECKPOINT), config=policy_cfg)
    policy.eval()
    model = policy.model
    model.eval()

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    num_steps = policy.config.num_inference_steps
    dt = -1.0 / num_steps
    x_t = noise.clone()
    trace_steps = []
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=x_t.device).expand(x_t.shape[0])
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, time_tensor)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)
        outputs_embeds, _ = model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -policy.config.chunk_size :].to(dtype=torch.float32)
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

    builtin = model.sample_actions(images, img_masks, tokens, masks, noise=noise.clone(), num_steps=num_steps)

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
        'prepared': {
            'observation.state': pack(prepared['observation.state']),
            'observation.images.left_wrist': pack(prepared['observation.images.left_wrist']),
            'observation.images.right_wrist': pack(prepared['observation.images.right_wrist']),
            'observation.images.right_high': pack(prepared['observation.images.right_high']),
        },
        'preprocessor_steps': preprocessor_steps,
        'model_inputs': {
            'images.cam_high': pack(images[0]),
            'images.cam_left_wrist': pack(images[1]),
            'images.cam_right_wrist': pack(images[2]),
            'img_masks.cam_high': pack(img_masks[0].to(torch.int32)),
            'img_masks.cam_left_wrist': pack(img_masks[1].to(torch.int32)),
            'img_masks.cam_right_wrist': pack(img_masks[2].to(torch.int32)),
            'tokens': pack(tokens),
            'token_masks': pack(masks.to(torch.int32)),
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
    (ROOT / 'evo_trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        'saved': str(ROOT / 'evo_trace.json'),
        'manual_builtin_equal': result['final']['manual_builtin_equal'],
        'manual_builtin_max_abs_diff': result['final']['manual_builtin_max_abs_diff'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
