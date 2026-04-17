# PI05 Non-DSRL Offline SAC Validation Checklist

目标：复用 `/data/cyy/RLinf/logs/20260416-22:38:11/demos` 这批真机 PI05 轨迹，逐步把 `openpi non-DSRL offline SAC` 接进 RLinf。

## 固定资源

- ReplayBuffer: `/data/cyy/RLinf/logs/20260416-22:38:11/demos`
- PI05 checkpoint: `/data/cyy/ckpts/adk111/piper_pi05_SFT_right_place_cup_on_book_left_insert_30000`
- 推荐解释器: `/data/cyy/RLinf/pi05_env/bin/python`

## Step 1: Replay 语义检查

目的：先确认 replay 里的 action / obs 到底是什么语义，再动 SAC。

命令：

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/check_replay_semantics.py
```

当前预期：

- `actions.shape[-1] == 700`
- `env_action_dim == 14`
- `action_chunk == 50`
- `curr_obs.states.shape[-1] == 14`
- `curr_obs.extra_view_images` 作为 wrist 输入
- `actions` 与 `forward_inputs.action` 一致或极接近

通过条件：

- 可以明确 offline actor/Q 应该学习的是 env action，而不是 DSRL latent noise
- 可以明确 actor 需要吃的观测字段

## Step 2: SAC 接口 smoke test

目的：在改代码前，先把“当前为什么不通”固化成一个可复现的检查。

命令：

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py
```

当前预期：

- 脚本应当直接通过
- `ForwardType.SAC` 和 `ForwardType.SAC_Q` 都能在真实 replay batch 上跑通

如果你需要回归早期失败状态：

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py --expect-failure=true
```

通过条件：

- `ForwardType.SAC` 返回 `sampled_actions, log_pi, shared_feature`
- `ForwardType.SAC_Q` 返回 `q_values`
- 输出 shape 与 replay action 语义一致

## Step 3: checkpoint reload 一致性

目的：保证新加的 actor/Q 模块保存再加载后输出不漂。

命令：

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py
```

当前预期：

- 同一输入下 reload 前后 `sampled_actions` 最大误差接近 0

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py --expect-failure=false
```

通过条件：

- 同一输入下 reload 前后 `sampled_actions` 最大误差接近 0

## Step 4: 单 batch 过拟合

目的：证明 loss 图是通的，不是只有接口能跑。

命令：

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/overfit_single_batch.py
```

当前预期：

- 这一步现在是骨架脚本
- 当 `sac_forward/sac_q_forward` 接通后，可以直接拿来测 loss 图

```bash
/data/cyy/RLinf/pi05_env/bin/python toolkits/openpi_non_dsrl_sac/overfit_single_batch.py --expect-failure=false --steps=20
```

通过条件：

- loss 有限
- 梯度非零
- 20 步内 loss 有下降趋势

## 建议的适配顺序

1. 先让 Step 2 和 Step 3 稳定通过
2. 再把 `overfit_single_batch.py` 里的占位 loss 换成真实 non-DSRL SAC loss
3. 验证小 batch 能学
4. 最后把配置接进 realworld offline runner

## 当前边界

- 这批脚本现在是“验证脚手架”，不是完整训练实现
- `overfit_single_batch.py` 里仍有 `TODO(agent)` 占位 loss，等 non-DSRL 路径接通后替换
