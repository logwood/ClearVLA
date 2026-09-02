# v76 原始代码状态重建参考

背景: normalized-residual 改动以无 flag 原地覆盖方式进入工作区, 且 v75 提交
(c339a01) 之后无任何 commit —— 正在跑的 v76 (E1-E8) 所执行的确切源码已不在
仓库任何提交中。本文档提供两条重建路径, 优先走 A。

## 路径 A（首选）: 训练机现场冻结

运行中的进程从训练机的某份代码启动。若该副本尚未同步 normalized-residual
编辑, 它就是 v76 的原件。在训练机上:

```bash
cd <训练机代码目录>
# 判别: 旧版特征是 _cross_update 中的
#   update = gate[:, None, None] * self.drop(out_proj(self._merge_heads(attended)))
# 且不存在 _normalize_residual 函数
grep -c '_normalize_residual' clearvla/experiments/observed_state_lab/policy_v39.py
# 输出 0 => 是 v76 原件, 立即封存:
tar czf ~/v76_source_exact.tgz clearvla scripts tests
```

即使已被同步覆盖, 也检查 `__pycache__/policy_v39.cpython-*.pyc` 的 mtime ——
若早于覆盖时间, 进程导入时的字节码仍在, 可作最后手段。

## 路径 B: 从当前树外科式回退（重建近似原件）

当前树 = v76 原件 + normalized-residual 改动(+31 行级别) + 配套仪表。
回退范围（全部在 policy_v39.py 的 hierarchical block 区域 + runtime console）:

1. 删除 `_sample_rms` / `_normalize_residual` / `_branch_geometry` 三个方法
   （`_branch_geometry` 若旧版已有简化版 serial_cancellation 计算, 恢复为
   forward 内联版, 见第 3 点）。
2. `_cross_update` 恢复为以下 verbatim 旧版（本文档作者于改动前逐行读取存档）:

```python
    def _cross_update(
        self,
        action: Tensor,
        memory: Tensor,
        *,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        mask: Tensor | None = None,
        value_gate: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_value = self._modulate(self.state_norm(action), shift, scale)
        q = self._split_heads(self.cross_q(query_value))
        key, value = kv_proj(self.condition_norm(memory)).chunk(2, dim=-1)
        k = self._split_heads(key)
        v = self._split_heads(value)
        if value_gate is not None:
            v = v * value_gate.to(device=v.device, dtype=v.dtype)[:, None, None, None]
        attended, weight = self._attention(q, k, v, mask)
        update = gate[:, None, None] * self.drop(out_proj(self._merge_heads(attended)))
        return action + update, update, weight
```

   注意旧版返回 3 元组（新版 6 元组）, 调用点需同步回退。
   旧版 value_gate 乘在注意力值上（归一化前语义）。
3. self/ffn 分支恢复直乘形式:
   `self_update = gates[:, 0, None, None] * self.drop(self.self_out(...))`
   `ffn_update = gates[:, 4, None, None] * self.drop(self.ffn(ffn_value))`
4. forward 的 metrics 块恢复旧键集（无 orthogonal_baseline / excess /
   realized_scale / weighted pair cosine; serial_cancellation 为
   `1 - |net| / sum(branch_norms)` 的内联计算）。
5. runtime console 移除 hmorth/hmxcan/realized_scale 键行。

回退结果与原件的差异仅在无关注释/空行级别; 打 tag `v76-reconstructed`
并注明"由 B 路径重建, 非逐位原件"。

## 事后固定动作

1. 当前树（含 normalized-residual）立即 commit: `v76.1-normalized-residual`。
2. A 或 B 的产物单独 commit + tag: `v76-exact` / `v76-reconstructed`。
3. 自本次起, 基座脚本 current_v48_justok.sh 已内置启动时代码快照
   (OUT_DIR/code_snapshot/: git_sha + uncommitted.diff + source.tgz,
   SNAPSHOT_CODE=0 可关) —— 本文档描述的事故类型从此结构性绝迹。
