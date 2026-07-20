# Qwen 航次语义证据

## 模块定位

Qwen 不生成轨迹，也不参与在线坐标回归。它仅将预测时刻已可见的 Destination、船型、尺度、吃水、ETA 和航行状态编码为冻结语义向量，为主航路意图提供软先验。

当前使用 `Qwen3-Embedding-0.6B`。离线编码遵循模型的检索格式：

- 输入为 `Instruct: {任务描述}\nQuery:{航次上下文}`；
- 使用左侧填充和末 token 池化；
- 输出向量进行 L2 归一化；
- 缺失上下文映射为全零向量。

## 逻辑链条

```text
预测时刻可见航次文本
        |
        v
Qwen3-Embedding-0.6B（冻结、离线）
        |
        v
轻量语义投影
        |
        v
与共享 Route Embedding 余弦对齐
        |
        v
主航路语义 logits -- 逐样本可靠性门控
        |                         |
        +----------+--------------+
                   v
      主航路后验（AIS 运动为主，语义为残差证据）
                   |
                   v
        置信度门控层级约束与子航路推断
                   |
                   v
         置信度感知 Top-1 / Top-K 路由
                   |
                   v
        航路条件候选轨迹生成与学习型选择
```

Qwen 对齐的 `Route Embedding` 与轨迹解码阶段使用同一组参数，因此该嵌入同时受到主航路分类、语义对齐和轨迹生成目标约束，不是孤立的文本支路。

默认不把 Qwen logits 直接注入子航路头。航次文本通常能提示大方向，但难以区分共享历史航段上的细小分支；子航路只通过主航路后验和层级约束间接受益，具体选择仍由 AIS 运动、航路原型、未来增强原型和可判别性共同完成。

可靠性门控同时读取运动特征和语义特征。训练时，以“语义证据相对非语义证据是否降低该样本的主航路交叉熵”为软目标校准门控；推理时不需要标签，仅由网络预测门值。文本缺失、过期或与运动冲突时，模型可降低语义权重并回退到 AIS 证据。

## 数据边界

侧车生成过程不读取主航路标签、子航路标签或真实未来轨迹。每个窗口只取得历史窗口结束时已有的航次上下文，因此侧车无标签且可跨固定划分复用。航路标签只在固定训练集内监督轻量投影器、可靠性门控和共享航路嵌入，验证与推理阶段不使用标签。

类别均衡辅助意图流默认不读取 Qwen 侧车。它只增强运动航路分类和意图表示，避免重采样分布改变语义先验；自然分布主训练流负责语义对齐、可靠性学习和轨迹回归。

## 生成侧车

在项目根目录运行：

```powershell
python utils\build_qwen_semantic_teacher.py `
  --context_path dataset\dma_raw_2023_06_07_08_plus_09_10_oa_s00\dma_2023_06_07_08_plus_09_10_oa_s00_target350_voyage_context.pkl `
  --model_path D:\Jason1982\wsl\Models\Qwen3-Embedding-0.6B `
  --output_path dataset\dma_raw_2023_06_07_08_plus_09_10_oa_s00\dma_2023_06_07_08_plus_09_10_oa_s00_target350_qwen3_embedding_0p6b.pkl `
  --batch_size 32 `
  --max_length 128 `
  --pooling last_token
```

主模型训练不加载 Qwen 权重，只读取离线侧车，因此显存和每轮耗时仅增加轻量投影及门控开销。

## 关键配置

```python
use_qwen_semantic_teacher = True
use_semantic_route_alignment = True
use_semantic_subroute_alignment = False
semantic_alignment_temperature = 0.20
semantic_route_alignment_weight = 0.05
semantic_subroute_alignment_weight = 0.0
semantic_route_reliability_weight = 0.05
semantic_reliability_temperature = 0.50
use_semantic_in_balanced_intent_stream = False
semantic_fusion_weight = 0.10
```

`use_semantic_subroute_alignment=True` 仅保留作“双层直接对齐”消融，不是推荐主模型。

## 实验验证

日志中的 `Semantic_Evidence` 报告：

- `route_acc`：仅看 Qwen 对齐 logits 的主航路准确率；
- `route_gate`：模型实际采用的平均语义权重；
- `target`：根据语义与非语义证据相对损失计算的平均软可靠性目标；
- `MAE`：门控预测与可靠性目标的平均绝对误差；
- 常规 `Route_ACC`、`Subroute_ACC`：融合全部运动、原型、层级约束与语义后的最终结果。

论文消融应在同一固定划分和随机种子下比较：主航路语义先验（推荐模型）、关闭 Qwen、以及双层直接语义对齐（失败或弱基线）。重点报告 ADE/FDE、分航路召回率、低置信度 Top-K 覆盖率和候选选择结果，而不是只比较总损失。
