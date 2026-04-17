# PI05 Non-DSRL Offline SAC Handoff

更新时间：2026-04-17
仓库路径：`/data/cyy/RLinf`
推荐环境：`/data/cyy/RLinf/pi05_env`

## 1. 当前任务

目标：在 **不破坏现有真机 PI05 推理/采集** 的前提下，给 RLinf 适配一条 **PI05 non-DSRL offline SAC** 训练路径，复用已经采集好的真机 ReplayBuffer 数据：

- ReplayBuffer 数据目录：`/data/cyy/RLinf/logs/20260416-22:38:11/demos`
- PI05 checkpoint：`/data/cyy/ckpts/adk111/piper_pi05_SFT_right_place_cup_on_book_left_insert_30000`

约束：

- **不能破坏** 现有命令：
  - `bash examples/embodiment/collect_data.sh realworld_piper_pi05_infer`
- 也就是不能改坏当前 realworld PI05 infer 的 rollout 推理链路。

---

## 2. 已确认的前提结论

### 2.1 现有 replay 数据的 action 语义

这批真机 PI05 replay 不是 DSRL latent noise，而是 **env action**。

已用脚本验证：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_replay_semantics.py
```

关键结果：

- `actions.shape[-1] = 700`
- `env_action_dim = 14`
- 可推得 `action_chunk = 50`
- `curr_obs.states.shape[-1] = 14`
- `curr_obs.extra_view_images` 被作为 wrist 输入
- `actions` 与 `forward_inputs.action` 完全一致：
  - `max_abs_diff = 0.0`
  - `mean_abs_diff = 0.0`

结论：

- offline actor/Q 需要学习的是 **14 维 env action × 50 chunk = 700 维展平动作**
- 不是 DSRL 的 32 维 latent noise

---

## 3. 这次新增的验证清单与脚手架

新增目录：

- `toolkits/openpi_non_dsrl_sac/`
- `tests/pi05_non_dsrl_sac/`

新增文件：

### 3.1 交互/验证脚本

- `toolkits/openpi_non_dsrl_sac/common.py`
  - replay 读取
  - obs/action 语义抽取
  - 最小 openpi cfg 构造
  - 张量统计 helper

- `toolkits/openpi_non_dsrl_sac/check_replay_semantics.py`
  - 检查 replay 数据语义是否和 non-DSRL SAC 设计一致

- `toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py`
  - 检查 `ForwardType.SAC` 和 `ForwardType.SAC_Q` 是否能在真实 replay batch 上跑通

- `toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py`
  - 检查保存/加载后 forward 是否一致

- `toolkits/openpi_non_dsrl_sac/overfit_single_batch.py`
  - 单 batch 过拟合骨架
  - 目前还是 placeholder loss，后续需要替换成真实 non-DSRL SAC loss

### 3.2 文档与测试

- `tests/pi05_non_dsrl_sac/README.md`
  - 整个适配步骤清单

- `tests/unit_tests/test_openpi_non_dsrl_sac_utils.py`
  - 轻量 helper 单测

已验证：

```bash
cd /data/cyy/RLinf/tests/unit_tests
/data/cyy/RLinf/pi05_env/bin/python -m pytest test_openpi_non_dsrl_sac_utils.py -q
```

结果：

- `4 passed`

---

## 4. 已做的核心代码修改

### 4.1 OpenPI 模型：新增 non-DSRL SAC 旁路

修改文件：

- `rlinf/models/embodiment/openpi/openpi_action_model.py`

这次的设计原则：

- **训练侧** 新增一条独立的 non-DSRL actor/Q side branch
- **推理侧** 保持原来的 diffusion `predict_action_batch()` 不动

也就是说：

- 没改 `predict_action_batch()` 的行为
- 没改 `sample_actions()` 的 rollout 推理语义
- 没改当前 realworld PI05 infer 使用的主路径

#### 新增配置字段

在 `OpenPi0Config` 里新增：

- `use_non_dsrl_sac: bool = False`
- `non_dsrl_state_dim: int = 14`
- `non_dsrl_num_images: int = 3`
- `non_dsrl_image_size: int = 64`
- `non_dsrl_image_latent_dim: int = 64`
- `non_dsrl_state_latent_dim: int = 64`
- `non_dsrl_hidden_dims: tuple = (256, 256, 256)`
- `non_dsrl_logstd_range: tuple = (-5.0, 2.0)`

#### 新增模块

当 `use_non_dsrl_sac=True` 时，初始化：

- `actor_image_encoder`
- `actor_state_encoder`
- `critic_image_encoder`
- `critic_state_encoder`
- `actor_trunk`
- `actor_mean`
- `actor_logstd`
- `q_head`

其中：

- image encoder / state encoder / multi-Q head 复用了已有模块：
  - `LightweightImageEncoder64`
  - `CompactStateEncoder`
  - `CompactMultiQHead`

#### 新增方法

新增 non-DSRL SAC 相关方法：

- `_normalize_sac_obs`
- `_select_wrist_source`
- `_stack_non_dsrl_images`
- `_preprocess_non_dsrl_states`
- `_encode_non_dsrl_actor_features`
- `_encode_non_dsrl_critic_features`
- `_sac_forward_non_dsrl`
- `_sac_q_forward_non_dsrl`

#### 当前 non-DSRL SAC 的 obs / action 设计

- obs：
  - `main_images`
  - `wrist_images` 或 `extra_view_images`
  - `states`
- 图像：
  - resize 到 `64x64`
  - 使用 3 路视角：`main + two wrist`
  - 值域归一到 `[-1, 1]`
- state：
  - flatten 成 `[B, 14]`
- actor 输出：
  - 展平动作 `[B, 700]`
  - reshape 成 `[B, 50, 14]`
- `log_pi`：
  - 当前返回逐维 `chunk_logprobs`，形状 `[B, 700]`
  - worker 会再 `sum(dim=-1, keepdim=True)`
- `SAC_Q`：
  - 输入展平动作 `[B, 700]`
  - 输出 `[B, num_q_heads]`

### 4.2 SAC worker：让 optimizer 分组识别 non-DSRL side branch

修改文件：

- `rlinf/workers/actor/fsdp_sac_policy_worker.py`

新增：

- `self.use_non_dsrl_sac = openpi_cfg.get("use_non_dsrl_sac", False)`

并把 critic optimizer 分组从：

- 只识别 `use_dsrl`

改成：

- `use_dsrl or use_non_dsrl_sac`

这样 critic 相关参数：

- `critic_image_encoder`
- `critic_state_encoder`
- `q_head`

会走 critic optimizer。

---

## 5. 已完成的验证结果

### 5.1 SAC 接口 smoke test：已通过

运行命令：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py
```

结果：`passed`

关键输出：

- `sampled_actions.shape = [2, 50, 14]`
- `log_pi.shape = [2, 700]`
- `q_values.shape = [2, 10]`

这说明：

- `ForwardType.SAC` 已经能在真实 replay batch 上跑通
- `ForwardType.SAC_Q` 也已经能跑通
- 已不再触发原先的：
  - `ValueError: sac_forward called but use_dsrl=False`

### 5.2 当前最重要的隔离结论

这次修改没有改动：

- `predict_action_batch()`
- `sample_actions()`
- `collect_data.sh realworld_piper_pi05_infer` 依赖的 rollout 推理路径

所以当前 non-DSRL SAC 是通过 **新增 side branch** 接入的，而不是替换原 PI05 推理逻辑。

---

## 6. 当前还没做完的部分

### 6.1 checkpoint reload 验证

脚本已经写好，但还没在这轮完整跑完：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py
```

目标：

- 确认同一输入下 reload 前后 `sampled_actions` 基本一致

### 6.2 单 batch 过拟合

脚本已经有骨架：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/overfit_single_batch.py --steps=20
```

但注意：

- 目前 `overfit_single_batch.py` 里的 loss 还是 placeholder
- 里面有 `TODO(agent)`
- 还没有替换成真实的 non-DSRL SAC actor/critic loss

### 6.3 还没有接 offline 训练配置

还没新增例如：

- `realworld_piper_pi05_non_dsrl_offline.yaml`

也还没有把 non-DSRL openpi 配置正式接到：

- `examples/embodiment/train_realworld_offline_async.sh`
- `examples/embodiment/train_realworld_offline_async.py`

---

## 7. 在 GPU 云服务器上建议继续做的顺序

### Step A. 先跑现有验证脚本

先确认代码迁到云端后，基础行为一致：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_replay_semantics.py
```

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py
```

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py
```

### Step B. 把 overfit 脚本里的 placeholder loss 替换成真实 SAC loss

当前脚本位置：

- `toolkits/openpi_non_dsrl_sac/overfit_single_batch.py`

需要做的事：

- 用真实 actor loss 替换 placeholder
- 用真实 critic loss 替换 placeholder
- 检查 20 步内 loss 是否下降

### Step C. 新建 offline 训练 YAML

建议新增：

- `examples/embodiment/config/realworld_piper_pi05_non_dsrl_offline.yaml`

建议从：

- `examples/embodiment/config/realworld_piper_resnet_5090.yaml`

和：

- `examples/embodiment/config/realworld_piper_pi05_infer.yaml`

拼出一份：

- replay 使用 `/data/cyy/RLinf/logs/.../demos`
- model 使用 `pi0_5`
- `actor.model.openpi.use_non_dsrl_sac: true`
- `actor.model.openpi.use_dsrl: false`
- `actor.model.state_dim: 14`
- `actor.model.action_dim: 14`
- `actor.model.num_action_chunks: 50`

然后再接：

```bash
bash examples/embodiment/train_realworld_offline_async.sh <new_config> <replay_dir>
```

### Step D. 做 shadow / offline eval，而不是直接真机

因为当前 non-DSRL actor/Q 还是新接入的：

- 先离线看 action 分布
- 再做 shadow 推理日志
- 最后再考虑是否接真机

---

## 8. 与现有真机 PI05 infer 相关的保护结论

这是这次最重要的要求，单独强调。

目前的修改：

- **没有改坏** 当前 `collect_data.sh realworld_piper_pi05_infer` 的主路径设计
- non-DSRL SAC 只在：
  - `openpi.use_non_dsrl_sac=True`
  - `ForwardType.SAC`
  - `ForwardType.SAC_Q`
  时才进入新逻辑
- 真实 rollout 仍走原来的：
  - `predict_action_batch()`
  - diffusion `sample_actions()`

换句话说：

- **训练路径新增**
- **推理路径保持不变**

---

## 9. 这次改过的关键文件

### 代码

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/workers/actor/fsdp_sac_policy_worker.py`

### 新增 toolkits

- `toolkits/openpi_non_dsrl_sac/__init__.py`
- `toolkits/openpi_non_dsrl_sac/common.py`
- `toolkits/openpi_non_dsrl_sac/check_replay_semantics.py`
- `toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py`
- `toolkits/openpi_non_dsrl_sac/check_openpi_checkpoint_reload.py`
- `toolkits/openpi_non_dsrl_sac/overfit_single_batch.py`

### 新增测试/文档

- `tests/pi05_non_dsrl_sac/README.md`
- `tests/unit_tests/test_openpi_non_dsrl_sac_utils.py`

---

## 10. 建议交接后第一条命令

到云服务器后，先跑这个：

```bash
PYTHONPATH=/data/cyy/RLinf /data/cyy/RLinf/pi05_env/bin/python \
toolkits/openpi_non_dsrl_sac/check_openpi_sac_interface.py
```

如果这条还能输出：

- `status: passed`
- `sampled_actions.shape = [B, 50, 14]`
- `q_values.shape = [B, 10]`

说明当前 non-DSRL 路径在新设备上至少基础接口是通的。

