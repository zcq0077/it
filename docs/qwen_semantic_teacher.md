# Qwen 离线语义教师

## 为什么改

旧方案让 Qwen 在主模型训练结束后对候选轨迹重新排序。验证结果表明，Qwen 的选择增益不稳定，最终经常被验证门控拒绝。根本原因是：当历史运动几乎相同时，Qwen 仅凭候选轨迹也不能凭空知道真实未来。

新方案不再让 Qwen 直接决定最终航路，而是把它用于更擅长的工作：读取预测时已经可见的 AIS 航次上下文，生成稳定的语义表示。主模型在每一折训练期间学习怎样使用这些表示。

## 数据流

```text
预测时可见的 AIS 上下文
  Destination、船型、船长、船宽、吃水、ETA、航行状态
                    |
                    v
             本地 Qwen3-1.7B
                    |
                    v
       离线 2048 维语义向量侧车文件
                    |
                    v
       轻量语义编码器 + 置信度门控融合
          |              |              |
          v              v              v
      主航路头        子航路头       候选选择器
                    |
                    v
               轨迹生成器
```

侧车生成过程不读取主航路标签、子航路标签和真实未来轨迹。每个轨迹窗口只会取得该窗口结束时已经存在的航次上下文；缺失上下文会映射到全零向量，模型自动退回纯运动特征。

## 一次性生成

在项目根目录和 `lucky` 环境中运行：

```powershell
python utils\build_qwen_semantic_teacher.py `
  --context_path dataset\dma_raw_2023_06_07_08\dma_voyage_context_2023_06_07_08.pkl `
  --model_path D:\Jason1982\wsl\Models\Qwen3-1.7B `
  --output_path dataset\dma_raw_2023_06_07_08\dma_qwen_semantic_teacher_v1.pkl `
  --batch_size 16 `
  --max_length 128
```

当前数据已经生成 `7507 x 2048` 的语义向量，文件约 29 MB。只要航次上下文文件和 Qwen 模型没有变化，就不需要重复生成。

## 训练

默认配置已经启用新方案，因此直接运行：

```powershell
python iTentformer.py
```

关键配置：

```python
use_qwen_semantic_teacher = True
qwen_semantic_path = "dataset/dma_raw_2023_06_07_08/dma_qwen_semantic_teacher_v1.pkl"
semantic_hidden_dim = 128
semantic_fusion_weight = 0.25
semantic_dropout = 0.20

# 关闭已经验证失败的训练后候选重排器，避免重复加载 Qwen。
use_qwen_reranker = False
```

训练时不会再次加载 1.7B Qwen，只读取约 29 MB 的语义侧车，并训练轻量语义编码器。因此每轮速度和显存开销只会小幅增加。

## 怎样判断是否有效

必须和同一数据、同一折、同一随机种子的纯轨迹版本比较：

```powershell
# 新语义教师版本
python iTentformer.py --model_prefix semantic_teacher_on

# 消融版本
python iTentformer.py --no-use_qwen_semantic_teacher --model_prefix semantic_teacher_off
```

重点比较验证集和测试集的 `ADE/FDE`、主航路/子航路 Top-1、稀有类召回率、候选 winner accuracy。不能只看总 loss，因为总 loss 还包含分类、可判别性、平滑和候选排序等辅助项。

新方案提高的是“有语义证据时的航路先验”，不能保证在历史与上下文都完全相同的情况下唯一判断未来。此时模型仍应保留多个候选，而不是伪造高置信度。
