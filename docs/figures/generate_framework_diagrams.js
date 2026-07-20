const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const outputDir = path.resolve(process.argv[2] || __dirname);
const galleryPath = process.argv[3] ? path.resolve(process.argv[3]) : null;
fs.mkdirSync(outputDir, { recursive: true });

const C = {
  paper: "#FFFFFF",
  ink: "#20262E",
  muted: "#5C6672",
  line: "#4B5563",
  lightLine: "#A7B0BA",
  backboneFill: "#EAF2FB",
  backboneStroke: "#4A79A8",
  proposedFill: "#ECF5EF",
  proposedStroke: "#3D7D55",
  priorFill: "#FFF7E6",
  priorStroke: "#9B6A20",
  trainFill: "#F4F0FA",
  trainStroke: "#7556A8",
  neutralFill: "#F7F8FA",
  neutralStroke: "#65707C",
};

const P = {
  backbone: [C.backboneFill, C.backboneStroke],
  proposed: [C.proposedFill, C.proposedStroke],
  prior: [C.priorFill, C.priorStroke],
  train: [C.trainFill, C.trainStroke],
  neutral: [C.neutralFill, C.neutralStroke],
};

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function start(width, height, title, description) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title>
  <desc id="desc">${esc(description)}</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,4 L0,8 z" fill="${C.line}"/>
    </marker>
    <marker id="arrowTrain" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,4 L0,8 z" fill="${C.trainStroke}"/>
    </marker>
    <style>
      text { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; fill: ${C.ink}; letter-spacing: 0; }
      .figure-title { font-size: 34px; font-weight: 500; }
      .group-title { font-size: 23px; font-weight: 500; }
      .box-title { font-size: 21px; font-weight: 500; }
      .body { font-size: 17px; font-weight: 400; }
      .small { font-size: 15px; font-weight: 400; fill: ${C.muted}; }
      .formula { font-size: 19px; font-weight: 500; }
      .step { font-size: 17px; font-weight: 500; }
    </style>
  </defs>
  <rect x="0" y="0" width="${width}" height="${height}" fill="${C.paper}"/>
  <text x="${width / 2}" y="52" text-anchor="middle" class="figure-title">${esc(title)}</text>
`;
}

function end() {
  return "</svg>\n";
}

function lines(x, y, values, options = {}) {
  const anchor = options.anchor || "middle";
  const cls = options.className || "body";
  const lineHeight = options.lineHeight || 26;
  const firstDy = options.firstDy || 0;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${values
    .map((value, index) => `<tspan x="${x}" dy="${index === 0 ? firstDy : lineHeight}">${esc(value)}</tspan>`)
    .join("")}</text>\n`;
}

function box(x, y, w, h, title, body = [], kind = "neutral", options = {}) {
  const [fill, stroke] = P[kind] || P.neutral;
  const dash = options.dashed ? ` stroke-dasharray="8 6"` : "";
  const titleY = y + (options.compact ? 29 : 35);
  const bodyY = options.bodyY || y + (options.compact ? 55 : 68);
  let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="2"${dash}/>`;
  s += lines(x + w / 2, titleY, [title], { className: "box-title" });
  if (body.length) {
    s += lines(x + w / 2, bodyY, body, {
      className: options.bodyClass || "body",
      lineHeight: options.lineHeight || 26,
    });
  }
  return s;
}

function group(x, y, w, h, title, kind = "neutral", options = {}) {
  const [, stroke] = P[kind] || P.neutral;
  const dash = options.dashed ? ` stroke-dasharray="10 7"` : "";
  const labelWidth = options.labelWidth || Math.max(150, title.length * 23 + 42);
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${C.paper}" stroke="${stroke}" stroke-width="2.2"${dash}/>
  <rect x="${x + 16}" y="${y - 17}" width="${labelWidth}" height="34" rx="4" fill="${C.paper}" stroke="${stroke}" stroke-width="1.8"${dash}/>
  <text x="${x + 16 + labelWidth / 2}" y="${y + 7}" text-anchor="middle" class="group-title">${esc(title)}</text>\n`;
}

function orth(points, options = {}) {
  const train = Boolean(options.train);
  const color = train ? C.trainStroke : (options.color || C.line);
  const dash = options.dashed || train ? ` stroke-dasharray="8 6"` : "";
  const marker = options.noArrow ? "" : ` marker-end="url(#${train ? "arrowTrain" : "arrow"})"`;
  const pointText = points.map(([x, y]) => `${x},${y}`).join(" ");
  return `<polyline points="${pointText}" fill="none" stroke="${color}" stroke-width="${options.width || 2.3}" stroke-linejoin="round"${dash}${marker}/>`;
}

function junction(x, y, kind = "neutral") {
  const [, stroke] = P[kind] || P.neutral;
  return `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="${stroke}"/>`;
}

function caption(x, y, text, options = {}) {
  const width = options.width || Math.max(70, text.length * 16 + 18);
  return `<rect x="${x - width / 2}" y="${y - 17}" width="${width}" height="23" rx="3" fill="${C.paper}"/>
  <text x="${x}" y="${y}" text-anchor="middle" class="small">${esc(text)}</text>\n`;
}

function mainArrow(x1, x2, y, label) {
  let s = orth([[x1, y], [x2, y]], { width: 3 });
  if (label) s += caption((x1 + x2) / 2, y - 12, label);
  return s;
}

function legend(x, y) {
  const items = [
    ["backbone", "原始主干"],
    ["proposed", "本文新增"],
    ["prior", "语义/几何先验"],
    ["train", "仅训练阶段"],
  ];
  let s = "";
  let cursor = x;
  for (const [kind, label] of items) {
    const [fill, stroke] = P[kind];
    s += `<rect x="${cursor}" y="${y}" width="22" height="15" rx="2" fill="${fill}" stroke="${stroke}" stroke-width="1.5"${kind === "train" ? ' stroke-dasharray="4 3"' : ""}/>`;
    s += lines(cursor + 31, y + 13, [label], { className: "small", anchor: "start" });
    cursor += kind === "prior" ? 170 : 145;
  }
  return s;
}

function overall() {
  const w = 3420;
  const h = 1650;
  let s = start(
    w,
    h,
    "改进 iTentformer 详细数据流总框架",
    "从历史船舶自动识别系统（AIS）数据、差分序列、航次语义和训练集航路原型开始，逐项展示运动编码、语义门控、主子航路分类得分融合、可判别路由、条件解码、残差合成、多候选生成和选择器输入。",
  ) + legend(1260, 72);

  s += group(35, 150, 290, 810, "输入与先验", "neutral");
  s += group(365, 150, 340, 810, "编码器", "backbone");
  s += group(745, 150, 1040, 810, "层级意图推理", "proposed");
  s += group(1825, 150, 660, 810, "条件轨迹解码", "proposed");
  s += group(2525, 150, 580, 810, "多候选与选择", "proposed");
  s += group(3145, 150, 235, 810, "最终输出", "neutral");

  s += box(60, 205, 240, 105, "状态序列 X", ["航向 COG、经度 Lon", "纬度 Lat、航速 SOG"], "neutral", { compact: true, lineHeight: 23 });
  s += box(60, 370, 240, 105, "差分序列 ΔX", ["航向/经纬度/航速的差分"], "neutral", { compact: true });
  s += box(60, 535, 240, 120, "预测前航次上下文 C", ["船型、吃水、目的港", "不读取真实未来"], "prior", { compact: true });
  s += box(60, 715, 240, 135, "训练集航路原型 R", ["主航路/子航路中心线", "仅由固定训练集构建"], "prior", { compact: true });

  s += box(395, 205, 280, 105, "Transformer 全局编码器", ["X → Hx（全局运动）"], "backbone", { compact: true });
  s += box(395, 370, 280, 105, "扩张时序卷积 TCN + SE", ["ΔX → Ht（局部意图序列）"], "backbone", { compact: true });
  s += box(395, 535, 280, 120, "千问语义编码器", ["C → 冻结语义向量", "多层感知机 MLP → Hq"], "prior", { compact: true });
  s += box(395, 715, 280, 135, "航路原型匹配评分", ["当前位置到中心线距离", "+ 历史方向与切向一致性", "→ 主/子航路原型得分"], "prior", { compact: true });
  s += orth([[300, 257], [395, 257]]);
  s += orth([[300, 422], [395, 422]]);
  s += orth([[300, 595], [395, 595]]);
  s += orth([[300, 782], [395, 782]]);

  s += box(780, 365, 200, 110, "意图特征摘要", ["历史均值 Mean(Ht)", "最后时刻 Last(Ht)", "首尾变化 Last−First"], "backbone", { compact: true, lineHeight: 23 });
  s += box(1020, 350, 230, 140, "门控语义融合", ["门值 g = sigmoid([Hi,Hq])", "Hf = Hi + 0.25·g·Hq", "输出融合特征 Hf"], "proposed", { compact: true, lineHeight: 25 });
  s += orth([[675, 422], [780, 422]], { label: "Ht" });
  s += orth([[980, 422], [1020, 422]]);
  s += orth([[675, 595], [740, 595], [740, 455], [1020, 455]]);
  s += caption(750, 575, "Hq", { width: 48 });

  s += box(1285, 205, 190, 105, "主航路分类头", ["Hf → 基础分类得分"], "proposed", { compact: true });
  s += box(1510, 195, 220, 125, "主航路得分融合 ⊕", ["基础得分 + 千问语义残差", "+ 主航路原型得分"], "proposed", { compact: true, lineHeight: 24 });
  s += box(1285, 565, 190, 105, "子航路分类头", ["Hf → 基础分类得分"], "proposed", { compact: true });
  s += box(1510, 535, 220, 165, "子航路得分融合 ⊕", ["基础得分 + 千问语义残差", "+ 子航路原型/未来先验", "+ 主航路概率 Pr 的层级偏置"], "proposed", { compact: true, lineHeight: 25 });

  s += orth([[1250, 420], [1265, 420], [1265, 257], [1285, 257]]);
  s += orth([[1250, 420], [1265, 420], [1265, 617], [1285, 617]]);
  s += orth([[1475, 257], [1510, 257]]);
  s += orth([[1475, 617], [1510, 617]]);

  s += box(1540, 345, 175, 115, "主航路置信路由", ["温度系数 = 1.30", "置信度 + 概率间隔 + dr"], "proposed", { compact: true, lineHeight: 24 });
  s += box(1540, 725, 175, 115, "子航路置信路由", ["温度系数 = 0.90", "置信度 + 概率间隔 + ds"], "proposed", { compact: true, lineHeight: 24 });
  s += orth([[1620, 320], [1620, 345]]);
  s += orth([[1620, 700], [1620, 725]]);
  s += box(1730, 345, 35, 115, "Pr", ["er"], "backbone", { compact: true, bodyClass: "small", lineHeight: 22 });
  s += box(1730, 725, 35, 115, "Ps", ["es"], "backbone", { compact: true, bodyClass: "small", lineHeight: 22 });
  s += orth([[1715, 402], [1730, 402]]);
  s += orth([[1715, 782], [1730, 782]]);
  s += orth([[1765, 402], [1775, 402], [1775, 515], [1620, 515], [1620, 535]]);
  s += caption(1710, 505, "主航路层级约束 Pr", { width: 130 });

  s += box(1860, 205, 270, 135, "局部意图条件注入 ⊕", ["Hc = Ht + Wr·er + Ws·es", "保留前2类时按概率软加权"], "proposed", { compact: true, lineHeight: 26 });
  s += box(1860, 420, 270, 105, "全局运动特征投影", ["Hx → 投影后的轨迹记忆特征"], "backbone", { compact: true });
  s += box(2170, 300, 125, 105, "特征拼接", ["[Hx || Hc]"], "neutral", { compact: true });
  s += box(2320, 285, 135, 135, "注意力融合块", ["四头注意力", "全局与局部特征交互"], "backbone", { compact: true, lineHeight: 24 });
  s += box(2170, 485, 285, 120, "共享轨迹解码器", ["线性映射 → 共享残差 Rshared", "学习跨航路共性残差"], "backbone", { compact: true });
  s += box(1860, 675, 210, 105, "线性运动基线", ["X → B(X)"], "backbone", { compact: true });
  s += box(2100, 675, 220, 135, "子航路残差专家库", ["停止梯度 stop-gradient(Hc)", "按 Ps 软加权/按类别选专家", "→ Rsub，缩放系数 = 0.25"], "proposed", { compact: true, lineHeight: 24 });
  s += box(2350, 675, 105, 135, "残差求和 ⊕", ["Y0 = B(X)", "+ 共享残差", "+ 子航路残差"], "neutral", { compact: true, lineHeight: 23 });

  s += orth([[1765, 402], [1810, 402], [1810, 250], [1860, 250]]);
  s += caption(1825, 390, "er", { width: 36 });
  s += orth([[1765, 782], [1810, 782], [1810, 295], [1860, 295]]);
  s += caption(1825, 760, "es", { width: 36 });
  s += orth([[2130, 272], [2150, 272], [2150, 335], [2170, 335]]);
  s += orth([[2130, 472], [2150, 472], [2150, 370], [2170, 370]]);
  s += orth([[2295, 352], [2320, 352]]);
  s += orth([[2387, 420], [2387, 485]]);
  s += orth([[2130, 272], [2150, 272], [2150, 675]]);
  s += orth([[2455, 545], [2470, 545], [2470, 715], [2455, 715]]);
  s += orth([[2070, 727], [2085, 727], [2085, 840], [2402, 840], [2402, 810]]);
  s += orth([[2320, 742], [2350, 742]]);

  s += box(2560, 205, 235, 120, "候选航路条件", ["全部 6 个子航路编号", "查表得到 er(k)、es(k)"], "proposed", { compact: true });
  s += box(2560, 390, 235, 150, "参数共享的条件轨迹解码", ["输入 Ht、Hx、er(k)、es(k)", "同一解码器 + 第 k 个专家", "→ Y1 ... Y6"], "proposed", { compact: true, lineHeight: 25 });
  s += box(2825, 390, 235, 150, "候选轨迹池", ["{Y0, Y1, ..., Y6}", "1 条基础轨迹 + 6 条分支", "共 7 条候选"], "proposed", { compact: true, lineHeight: 25 });
  s += box(2560, 675, 235, 155, "选择器输入特征拼接", ["Hf || 主/子航路嵌入", "|| 主/子航路对数概率", "|| 连续性/平滑性", "|| 原型距离/基础候选标记"], "prior", { compact: true, lineHeight: 24 });
  s += box(2825, 675, 235, 155, "候选轨迹选择器", ["层归一化 → MLP → 候选得分", "+ 航路概率先验", "概率归一化 + 校准切换"], "proposed", { compact: true, lineHeight: 25 });

  s += orth([[2677, 325], [2677, 390]]);
  s += orth([[2455, 742], [2485, 742], [2485, 870], [2942, 870], [2942, 540]]);
  s += caption(2700, 858, "基础轨迹 Y0", { width: 82 });
  s += orth([[2795, 465], [2825, 465]]);
  s += orth([[2942, 540], [2942, 650], [2677, 650], [2677, 675]]);
  s += orth([[2795, 752], [2825, 752]]);
  s += orth([[3060, 752], [3110, 752], [3110, 510], [3145, 510]]);

  s += box(3175, 360, 175, 300, "最终轨迹 Ŷ", ["未来 12 点 / 3 小时", "经度 Lon、纬度 Lat", "航向 COG、航速 SOG", "选中的主航路", "选中的子航路", "置信度", "不确定性"], "neutral", { lineHeight: 29 });

  s += group(365, 1110, 2740, 390, "训练阶段监督（推理时全部移除）", "train", { dashed: true, labelWidth: 350 });
  s += box(410, 1170, 580, 240, "标签与小类均衡意图流", ["主/子航路标签 + 分阶段可判别性", "轨迹级均衡采样 + 类别权重", "主航路/子航路/可判别性/对比损失", "只更新意图相关模块，不替换自然分布主流"], "train", { dashed: true, lineHeight: 30 });
  s += box(1040, 1170, 580, 240, "真实未来轨迹监督", ["真实未来 Ygt → 轨迹回归监督", "均方误差 + 地理距离 + 终点位移误差 FDE", "+ 平滑损失 + 环形航向 COG 损失", "验证：平均位移误差 ADE + 0.2·FDE"], "train", { dashed: true, lineHeight: 30 });
  s += box(1670, 1170, 580, 240, "未来增强意图教师", ["未来相对位移 → 未来运动编码器", "形成子航路未来模式原型", "仅在可判别窗口对齐 Hf", "教师分类得分作为温和残差"], "train", { dashed: true, lineHeight: 30 });
  s += box(2300, 1170, 755, 240, "候选优胜者监督", ["候选代价 = 平均位移误差 ADE + 0.2·终点误差 FDE", "取最小代价候选作为优胜目标", "选择器损失 = 软排序 + 代价回归", "候选损失 = 优胜轨迹回归；推理时不看真实未来"], "train", { dashed: true, lineHeight: 30 });
  s += orth([[700, 1170], [700, 1010], [1380, 1010], [1380, 670]], { train: true });
  s += orth([[700, 1010], [1240, 1010], [1240, 257], [1285, 257]], { train: true });
  s += orth([[1330, 1170], [1330, 1040], [2387, 1040], [2387, 810]], { train: true });
  s += orth([[1960, 1170], [1960, 1060], [1490, 1060], [1490, 675], [1510, 675]], { train: true });
  s += orth([[2677, 1170], [2677, 1080], [2942, 1080], [2942, 830]], { train: true });
  s += lines(1710, 1570, ["实线：训练与推理共享的数据流；紫色虚线：仅训练监督；所有融合节点均在图中显式标注。"], { className: "small" });
  return s + end();
}

function intent() {
  const w = 2200;
  const h = 1120;
  let s = start(w, h, "层级航路意图与可判别性模块", "运动历史、千问语义与航路中心线先验形成主航路和子航路概率，并通过可判别性门控决定唯一选择或保留前两类软分布。") + legend(700, 72);

  s += group(45, 155, 360, 760, "可观测输入", "neutral");
  s += box(75, 220, 300, 145, "历史运动 H", ["TCN 历史摘要", "均值 + 最后时刻 + 首尾变化", "无真实未来"], "backbone");
  s += box(75, 445, 300, 145, "千问语义 Hq", ["船型 / 吃水 / 目的港", "冻结、离线、不含航路标签"], "prior");
  s += box(75, 670, 300, 155, "航路几何原型", ["训练集中心线", "距离 + 切向方向", "主类与子类分别构建"], "prior");

  s += group(465, 155, 430, 760, "意图特征构建", "proposed");
  s += box(505, 220, 350, 170, "门控语义融合", ["Hq → 语义编码器", "门值 g = sigmoid([H, Hq])", "Hf = H + 0.25·g·Hq"], "proposed");
  s += box(505, 465, 350, 145, "主航路分类头", ["主航路分类得分", "+ 千问语义残差", "+ 主航路原型先验"], "proposed");
  s += box(505, 690, 350, 155, "子航路分类头", ["子航路分类得分", "+ 主子航路层级约束", "+ 子航路/未来模式先验"], "proposed");
  s += orth([[680, 390], [680, 465]]);
  s += orth([[680, 610], [680, 690]]);

  s += group(955, 155, 450, 760, "可靠性判断", "proposed");
  s += box(995, 220, 370, 175, "主航路可判别门", ["最高类别置信度", "第一名与第二名概率间隔", "学习得到的可判别性 dr", "三项共同通过才允许唯一选择"], "proposed");
  s += box(995, 470, 370, 135, "置信度门控层级约束", ["大类可靠：加强父子约束", "大类不可靠：自动减弱约束"], "proposed");
  s += box(995, 680, 370, 175, "子航路可判别门", ["置信度 + 概率间隔 + ds", "可判别：唯一子航路", "不可判别：保留前两类", "避免分叉前强行猜测"], "proposed");
  s += orth([[1180, 395], [1180, 470]]);
  s += orth([[1180, 605], [1180, 680]]);

  s += group(1465, 155, 690, 760, "概率条件输出", "backbone");
  s += box(1510, 230, 600, 160, "主航路概率 P(主航路)", ["OA / OB1 / OB2 / OC", "温度系数 = 1.30", "高置信时唯一选择，否则保留校准后的前两类"], "backbone");
  s += box(1510, 485, 600, 180, "子航路概率 P(子航路)", ["OA_S00 / OA_S01 / OA_S02", "OB1_S00 / OB2_S00 / OC_S00", "温度系数 = 0.90", "高置信时唯一选择，否则保留校准后的前两类"], "backbone");
  s += box(1510, 745, 600, 110, "条件回注", ["主航路概率 → 主航路嵌入；子航路概率 → 子航路嵌入"], "backbone");
  s += orth([[1810, 390], [1810, 485]]);
  s += orth([[1810, 665], [1810, 745]]);

  s += orth([[405, 292], [465, 292]]);
  s += orth([[405, 517], [435, 517], [435, 330], [505, 330]]);
  s += orth([[405, 747], [435, 747], [435, 535], [505, 535]]);
  s += orth([[405, 747], [450, 747], [450, 767], [505, 767]]);
  s += orth([[895, 535], [955, 535]]);
  s += orth([[895, 767], [955, 767]]);
  s += orth([[1405, 307], [1465, 307]]);
  s += orth([[1405, 767], [1465, 767]]);

  s += group(465, 985, 940, 85, "训练监督", "train", { dashed: true, labelWidth: 160 });
  s += lines(935, 1030, ["分阶段硬/软标签 + 可判别性二元交叉熵 + 难样本聚焦/类别权重 + 监督式对比损失"], { className: "body" });
  s += orth([[680, 985], [680, 915]], { train: true });
  s += orth([[1180, 985], [1180, 915]], { train: true });
  return s + end();
}

function candidates() {
  const w = 2280;
  const h = 1110;
  let s = start(w, h, "多候选轨迹生成与学习式筛选模块", "基础预测与六条子航路条件预测组成候选集合，候选选择器使用历史、航路概率和几何合理性选出唯一轨迹。") + legend(740, 72);

  s += box(55, 235, 280, 175, "历史编码 H", ["TCN 局部时序编码", "+ Transformer 全局编码", "千问语义已经门控融合", "主/子航路分类得分"], "backbone", { lineHeight: 24 });
  s += box(55, 520, 280, 175, "候选标签集合", ["基础轨迹", "OA_S00 / S01 / S02", "OB1_S00 / OB2_S00 / OC_S00"], "proposed");

  s += group(405, 155, 470, 650, "共享条件解码器", "proposed");
  s += box(450, 220, 380, 125, "条件注入", ["H + 第 k 个主航路嵌入 + 第 k 个子航路嵌入"], "proposed");
  s += box(450, 420, 380, 125, "共享轨迹解码", ["Transformer + 注意力融合", "线性运动基线 + 共享残差"], "backbone");
  s += box(450, 620, 380, 125, "分支残差修正", ["第 k 个专家残差，缩放系数 = 0.25"], "proposed");
  s += orth([[640, 345], [640, 420]]);
  s += orth([[640, 545], [640, 620]]);

  s += group(945, 155, 500, 650, "候选轨迹集合", "proposed");
  const labels = ["基础轨迹", "OA-S00", "OA-S01", "OA-S02", "OB1-S00", "OB2-S00", "OC-S00"];
  for (let i = 0; i < labels.length; i++) {
    s += box(990, 205 + i * 78, 410, 55, `Ŷ${i}  ${labels[i]}`, [], i === 0 ? "backbone" : "proposed", { compact: true });
  }

  s += group(1515, 155, 430, 650, "候选评分", "proposed");
  s += box(1555, 220, 350, 230, "六维几何与概率特征", ["第 k 个主航路的对数概率", "第 k 个子航路的对数概率", "连续性 / 平滑性", "原型距离 / 基础候选标记"], "prior", { lineHeight: 31 });
  s += box(1555, 535, 350, 180, "候选轨迹选择器", ["历史 H、航路嵌入、六维评分特征", "层归一化 → MLP → 第 k 个候选得分", "+ 航路概率先验"], "proposed");
  s += orth([[1730, 450], [1730, 535]]);

  s += box(2015, 315, 220, 280, "唯一预测 Ŷ", ["候选概率归一化", "置信度阈值", "前两候选得分间隔", "验证集校准", "控制是否切换分支"], "neutral", { lineHeight: 31 });

  s += orth([[335, 322], [405, 322]]);
  s += orth([[335, 607], [370, 607], [370, 280], [450, 280]]);
  s += orth([[875, 480], [945, 480]]);
  s += orth([[1445, 480], [1515, 480]]);
  s += orth([[1945, 625], [1980, 625], [1980, 455], [2015, 455]]);

  s += group(405, 920, 1540, 125, "仅训练阶段：候选优胜者监督", "train", { dashed: true, labelWidth: 320 });
  s += box(455, 950, 290, 65, "真实未来 Y", [], "train", { dashed: true, compact: true });
  s += box(850, 950, 420, 65, "候选代价 = 平均误差 ADE + 0.2·终点误差 FDE", [], "train", { dashed: true, compact: true });
  s += box(1375, 950, 520, 65, "软排序 + 代价回归 + 优胜候选损失", [], "train", { dashed: true, compact: true });
  s += orth([[745, 982], [850, 982]], { train: true });
  s += orth([[1270, 982], [1375, 982]], { train: true });
  s += orth([[1635, 950], [1635, 805]], { train: true });
  return s + end();
}

function experts() {
  const w = 2180;
  const h = 1080;
  let s = start(w, h, "通用子航路残差专家模块", "运动基线、共享轨迹残差与子航路专家残差沿三条平行通道计算，最后通过统一求和节点合成预测。") + legend(720, 72);

  s += lines(60, 155, ["并行残差通道"], { className: "group-title", anchor: "start" });
  s += orth([[50, 175], [2100, 175]], { noArrow: true, color: C.lightLine, width: 1.5 });

  s += box(60, 225, 280, 140, "历史状态 X", ["最后位置与速度趋势", "未来 12 步"], "neutral");
  s += box(430, 225, 380, 140, "运动基线 B(X)", ["残差式线性外推", "保持位置与速度连续"], "backbone");

  s += box(60, 465, 280, 140, "运动表征 H", ["TCN 局部时序编码", "+ Transformer 全局编码", "共享历史特征"], "backbone", { lineHeight: 24 });
  s += box(430, 465, 380, 140, "共享残差 Rshared", ["注意力融合 + 轨迹解码器", "学习跨航路共性运动"], "backbone");

  s += box(60, 705, 280, 140, "子航路条件", ["子航路概率或类别编号", "主/子航路嵌入向量"], "proposed");
  s += box(430, 705, 270, 140, "停止梯度", ["阻断专家私有梯度", "保护共享表示"], "proposed");
  s += group(785, 665, 760, 220, "子航路残差专家库", "proposed");
  const names = ["OA_S00", "OA_S01", "OA_S02", "OB1_S00", "OB2_S00", "OC_S00"];
  for (let i = 0; i < names.length; i++) {
    const row = Math.floor(i / 3);
    const col = i % 3;
    s += box(820 + col * 235, 710 + row * 80, 200, 55, names[i], [], "proposed", { compact: true });
  }
  s += lines(1165, 872, ["每个专家：层归一化 LN → 多层感知机 MLP → 子航路修正 ΔYk；末层零初始化"], { className: "small" });

  s += box(1625, 680, 280, 175, "专家路由", ["普通前向：按子航路概率软加权", "候选前向：选第 k 个类别专家", "统一乘缩放系数 0.25"], "proposed");

  s += orth([[340, 295], [430, 295]]);
  s += orth([[340, 535], [430, 535]]);
  s += orth([[340, 775], [430, 775]]);
  s += orth([[700, 775], [785, 775]]);
  s += orth([[1545, 775], [1625, 775]]);

  const busX = 1970;
  s += orth([[810, 295], [busX, 295]], { noArrow: true });
  s += orth([[810, 535], [busX, 535]], { noArrow: true });
  s += orth([[1905, 775], [busX, 775]], { noArrow: true });
  s += orth([[busX, 295], [busX, 775]], { noArrow: true });
  s += junction(busX, 295);
  s += junction(busX, 535);
  s += junction(busX, 775);
  s += box(2015, 430, 120, 210, "求和", ["Ŷ =", "B(X)", "+ Rshared", "+ Rsub"], "neutral", { lineHeight: 30 });
  s += orth([[busX, 535], [2015, 535]]);

  s += box(430, 950, 1475, 70, "设计原则", ["共享主干学习通用运动；每个专家只负责小幅、可控、与数据来源无关的子航路修正。"], "neutral", { compact: true });
  return s + end();
}

function training() {
  const w = 2300;
  const h = 1120;
  let s = start(w, h, "双流训练与联合损失框架", "自然分布主流训练完整轨迹任务，小类均衡辅助流仅训练意图任务，训练专用未来教师和候选监督共同汇入联合损失。") + legend(750, 72);

  s += group(45, 155, 1520, 750, "并行训练数据流", "neutral");
  s += box(80, 235, 260, 155, "固定训练窗口", ["按船舶唯一识别码 MMSI", "固定划分训练/验证/测试", "3 小时历史 → 3 小时未来", "保持自然类别分布"], "neutral", { lineHeight: 24 });
  s += box(430, 235, 340, 155, "自然分布主训练流", ["完整 iTentformer 前向", "意图 + 解码 + 候选", "保护总体平均/终点位移误差"], "backbone");
  s += box(860, 215, 650, 195, "完整轨迹目标 Ltraj", ["自动加权：意图均方误差 + 轨迹均方误差", "+ 球面地理距离 + 终点位移误差 FDE", "+ 轨迹平滑损失 + 环形航向 COG 损失"], "backbone", { lineHeight: 31 });
  s += orth([[340, 312], [430, 312]]);
  s += orth([[770, 312], [860, 312]]);

  s += box(80, 580, 260, 155, "轨迹级均衡采样", ["逆频次温和加权", "20% 额外编码", "避免长轨迹窗口霸占"], "proposed");
  s += box(430, 580, 340, 155, "小类意图辅助流", ["仅意图模式前向", "不运行轨迹解码器", "不替换自然分布窗口"], "proposed");
  s += box(860, 560, 650, 195, "意图目标 Lintent", ["主航路交叉熵 + 子航路难样本聚焦损失", "+ 类别权重 + 分阶段软标签", "+ 可判别性二元交叉熵 + 对比损失"], "proposed", { lineHeight: 31 });
  s += orth([[340, 657], [430, 657]]);
  s += orth([[770, 657], [860, 657]]);

  s += group(1625, 155, 630, 750, "训练专用监督", "train", { dashed: true, labelWidth: 230 });
  s += box(1670, 225, 540, 180, "未来增强意图教师", ["真实未来相对位移 → 未来运动编码器", "形成子航路未来模式原型", "仅可判别历史参与对齐", "推理阶段完全移除"], "train", { dashed: true });
  s += box(1670, 535, 540, 180, "候选优胜者监督", ["候选轨迹与真实未来比较", "候选代价 = 平均误差 ADE + 0.2·终点误差 FDE", "软排序损失 + 代价回归损失", "推理改用学习式选择器"], "train", { dashed: true });

  s += group(500, 990, 1300, 85, "联合优化", "proposed", { labelWidth: 160 });
  s += lines(1150, 1036, ["L = Ltraj + Lintent + λfuture·Lfuture + λselector·Lselector + λcandidate·Lcandidate"], { className: "formula" });

  const lossBusY = 940;
  s += orth([[1510, 312], [1535, 312], [1535, lossBusY]], { noArrow: true });
  s += orth([[1510, 657], [1535, 657]], { noArrow: true });
  s += orth([[2210, 315], [2230, 315], [2230, lossBusY]], { noArrow: true, train: true });
  s += orth([[2210, 625], [2230, 625]], { noArrow: true, train: true });
  s += orth([[1150, lossBusY], [2230, lossBusY]], { noArrow: true });
  s += junction(1535, lossBusY, "proposed");
  s += junction(2230, lossBusY, "train");
  s += orth([[1150, lossBusY], [1150, 990]], { color: C.proposedStroke });
  s += lines(2250, 1055, ["验证集：平均误差 ADE + 0.2·终点误差 FDE", "早停耐心值 = 10"], { className: "small", anchor: "end", lineHeight: 23 });
  return s + end();
}

const figures = [
  ["fig1-overall-framework", overall(), "总框架图"],
  ["fig2-hierarchical-intent", intent(), "层级意图与可判别性"],
  ["fig3-candidate-selector", candidates(), "多候选生成与选择"],
  ["fig4-subroute-residual-experts", experts(), "子航路残差专家"],
  ["fig5-training-objectives", training(), "双流训练与联合损失"],
];

async function writeAll() {
  const gallery = [];
  for (const [name, svg, label] of figures) {
    fs.writeFileSync(path.join(outputDir, `${name}.svg`), svg, "utf8");
    await sharp(Buffer.from(svg), { density: 180 })
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .withMetadata({ density: 300 })
      .toFile(path.join(outputDir, `${name}.png`));
    gallery.push({ label, svg });
  }

  if (galleryPath) {
    fs.mkdirSync(path.dirname(galleryPath), { recursive: true });
    const sources = gallery.map((item) => `data:image/svg+xml;base64,${Buffer.from(item.svg, "utf8").toString("base64")}`);
    const buttons = gallery.map((item, index) => `<button type="button" class="btn${index === 0 ? " btn-primary" : ""}" data-index="${index}" aria-pressed="${index === 0}">${item.label}</button>`).join("\n");
    const fragment = `<div id="itentformer-paper-framework">
  <div class="viz-controls" role="group" aria-label="选择框架图">
    ${buttons}
  </div>
  <div style="margin-top: 12px; width: 100%; overflow: hidden;">
    <img id="itentformer-paper-image" src="${sources[0]}" alt="${gallery[0].label}" style="display:block; width:100%; height:auto; border:1px solid var(--border);" />
  </div>
  <script>
    (() => {
      const root = document.getElementById("itentformer-paper-framework");
      const image = root.querySelector("#itentformer-paper-image");
      const sources = ${JSON.stringify(sources)};
      const labels = ${JSON.stringify(gallery.map((item) => item.label))};
      root.querySelectorAll("button[data-index]").forEach((button) => {
        button.addEventListener("click", () => {
          const index = Number(button.dataset.index);
          image.src = sources[index];
          image.alt = labels[index];
          root.querySelectorAll("button[data-index]").forEach((peer) => {
            const selected = peer === button;
            peer.setAttribute("aria-pressed", String(selected));
            peer.classList.toggle("btn-primary", selected);
          });
        });
      });
    })();
  </script>
</div>
`;
    fs.writeFileSync(galleryPath, fragment, "utf8");
  }
}

writeAll().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
