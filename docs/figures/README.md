# iTentformer 论文框架图

本目录按当前 `config_iTentformer.py` 与 `model.py` 的实际启用流程生成论文式框架图。图版采用深度学习论文常见的左到右主流程、正交箭头、分组边界与训练/推理解耦布局。

| 图号 | 内容 | PNG | 可编辑矢量图 |
|---|---|---|---|
| Fig. 1 | 改进 iTentformer 详细数据流总框架（含张量名、融合节点与模块输入） | `fig1-overall-framework.png` | `fig1-overall-framework.svg` |
| Fig. 2 | 层级航路意图与可判别性 | `fig2-hierarchical-intent.png` | `fig2-hierarchical-intent.svg` |
| Fig. 3 | 多候选轨迹生成与学习式筛选 | `fig3-candidate-selector.png` | `fig3-candidate-selector.svg` |
| Fig. 4 | 通用子航路残差专家 | `fig4-subroute-residual-experts.png` | `fig4-subroute-residual-experts.svg` |
| Fig. 5 | 双流训练与联合损失 | `fig5-training-objectives.png` | `fig5-training-objectives.svg` |

图中蓝色表示原始 iTentformer 主干，绿色表示本文新增模块，黄色表示 Qwen 语义或航路几何先验，紫色虚线表示仅在训练阶段使用的监督。总图显式标出了 `X / ΔX / Hx / Ht / Hq / Hf / Pr / Ps / er / es`，以及 Gate、Logit Fusion、Concat、FusionBlock、残差 Sum 和 Selector Feature Concat 等融合位置。

建议论文正文优先插入 SVG；Word 不兼容时使用对应高清 PNG。图中的英文模块名、符号和颜色可在 `generate_framework_diagrams.js` 中统一修改后重新生成。
