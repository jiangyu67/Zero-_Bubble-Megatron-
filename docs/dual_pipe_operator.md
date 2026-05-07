# DualPipe（`--pipeline-schedule dual_pipe`）— 运维说明

## 当前版本打开了什么

- **仅支持稠密（dense）流水线并行**（本 MVP 未接 MoE / DeepEP）。
- **`--pipeline-schedule dual_pipe`** 要求：**`--pipeline-model-parallel-size` > 1**、**不能开虚拟流水线（VPP）**；且与 **Torch FSDP2**、**Megatron FSDP**、**meta 初始化**、以及 **本地 CUDA 图 + full-iteration** 组合**不兼容**；当前实现还要求 **关闭 `--overlap-grad-reduce`**（副副本梯度在迭代末一次性并入主副本 `main_grad`，尚未与异步 DDP hook 对齐）。
- 训练流程会在主 DDP 块上再挂一份**同拓扑的「第二套权重视图」**：以 **`Float16Module`** 形式存在，**不进入 optimizer 的 `model` 列表**；每次 **`optimizer.step()`** 之后（以及 **checkpoint 只加载到主副本** 之后），会用 **`copy_` 把主副本权重同步到这份副副本**。

## 调度行为（真·排程：在 1F1B 骨架上做 replica 路由）

- **P2P / warmup / steady / cooldown** 与 **`no_interleaving` 的 1F1B** 相同；差别在于 **每个全局 microbatch 的前向用哪套权重**：
  - **偶数** microbatch id → **主副本（DDP）**；
  - **奇数** id → **副副本（`Float16Module`，无 DDP）**。
- 反向仍由 autograd 在**各自前向建好的图**上完成；奇数 microbatch 的梯度落在副副本参数上，在进入 **`finalize_model_grads`** 之前会执行 **`merge_dual_pipe_bwd_grads_into_fwd`**，把 **`param.grad` 累加到主副本的 `main_grad`**，再走原有 DP/TP/LN 等归约。
- **与论文完全一致的「双向排程表 + 更激进的 P2P 重叠」**仍可继续迭代（例如按 PP rank 定制 microbatch 顺序、NVTX 细粒度区间）；当前版本先落实 **「双权重 + 奇偶条带 + 梯度合并」** 这一可运行子集。

## 显存与 profiling

- 每个 PP rank 上参数显存大约会**多出一套 BF16/FP16 权重**（再加框架开销），请按**约双倍参数显存**预估。
- 在 **PP ≥ 4** 上做重叠验证时，可沿用现有的 **`--profile` / NVTX**；并与同拓扑下的 **`no_interleaving`** 对照。

## FP8 / TE

- 若与 **FP8/FP4** 同时启用，`dual_pipe` 会在 **rank 0** 打印**实验性组合**警告；请相对 **BF16 基线**核对 loss / grad norm。

---

## 为什么要复制「双份」权重副本？

核心原因来自 **DualPipe / 双向流水** 的设计目标，而不是为了「多占显存好玩」：

1. **前向与反向可以同时在时间轴上推进**  
   前向沿流水线一个方向传激活，反向沿**相反方向**传梯度；在同一 PP stage 上，若仍只有**一套**正在参与 autograd 的参数，很容易出现：**同一套参数既要服务当前 microbatch 的前向图，又要服务更早 microbatch 的反向图**——在实现上要么串行掉重叠，要么在参数读写上与 autograd 规则打架。

2. **用两套参数视图把「前向用的权」和「反向用的权」拆开**  
   论文里常见的做法是维护 **fwd 副本** 与 **bwd 副本**（或等价的两套视图）：让不同方向的计算落在不同存储上，才能在做 **P2P 与计算重叠** 时，不把同一块显存同时当成「还在反传的叶子」和「已经用于下一轮前向」的战场。

3. **优化器仍然只能更新「一份逻辑参数」**  
   采用 **主从**：**只有主副本进 DDP + optimizer**；每个迭代里，副副本上前向产生的梯度在 **`finalize_model_grads` 之前** 并入主副本 **`main_grad`**；**`optimizer.step()` 之后** 再把主权重 **`copy_` 到副副本**，保证下一步两套权重仍一致。

4. **`defer_embedding_wgrad_compute`**  
   与「最后一级可能在主/副两套 embedding 上攒激活」的交互尚未支持；若开启该选项，**`dual_pipe` 会直接报错**，请关闭或改用 `no_interleaving`。

简言之：**双份副本**给不同 microbatch 的前向提供**独立参数存储**，为以后更激进的 **双向流水 + P2P 重叠**打基础；**每步末主→从的权重复制** + **迭代末副→主的梯度合并** 保证 **单优化器、单套逻辑参数** 下训练闭环正确。
