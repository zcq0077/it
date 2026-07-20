const fs = require("fs");
const path = require("path");
let sharp = null;
try {
  sharp = require("sharp");
} catch (_) {
  // SVG generation is dependency-free; PNG can be rendered by a headless browser.
}

const W = 3200;
const H = 1800;
const OUT_DIR = __dirname;
const SVG_PATH = path.join(OUT_DIR, "fig6-sci-vessel-trajectory-framework-16x9.svg");
const PNG_PATH = path.join(OUT_DIR, "fig6-sci-vessel-trajectory-framework-16x9.png");

const C = {
  paper: "#FFFFFF",
  ink: "#1F2937",
  muted: "#5F6B78",
  line: "#4B5563",
  border: "#C8D0D8",
  blue: "#377EB8",
  blueFill: "#EAF3FA",
  teal: "#009E73",
  tealFill: "#E7F6F1",
  green: "#4DAF4A",
  greenDark: "#2E7D32",
  greenFill: "#EDF7ED",
  orange: "#E69F00",
  orangeDark: "#A86500",
  orangeFill: "#FFF4DE",
  red: "#8B1E2D",
  redFill: "#FCECEE",
  gray: "#6B7280",
  grayFill: "#F3F4F6",
  grayBand: "#F7F8FA",
};

const themes = {
  blue: [C.blueFill, C.blue],
  teal: [C.tealFill, C.teal],
  green: [C.greenFill, C.greenDark],
  orange: [C.orangeFill, C.orangeDark],
  red: [C.redFill, C.red],
  gray: [C.grayFill, C.gray],
  white: [C.paper, C.border],
};

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function textLines(x, y, values, options = {}) {
  const list = Array.isArray(values) ? values : [values];
  const anchor = options.anchor || "middle";
  const cls = options.cls || "body";
  const lineHeight = options.lineHeight || 29;
  const fill = options.fill ? ` fill="${options.fill}"` : "";
  const weight = options.weight ? ` font-weight="${options.weight}"` : "";
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}"${fill}${weight}>${list
    .map((value, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${esc(value)}</tspan>`)
    .join("")}</text>`;
}

function titleBlock(x, y, w, lines, options = {}) {
  const list = Array.isArray(lines) ? lines : [lines];
  const lineHeight = options.lineHeight || 28;
  const fontSize = options.fontSize || 25;
  return `<text x="${x + w / 2}" y="${y}" text-anchor="middle" font-size="${fontSize}" font-weight="700">${list
    .map((value, index) => `<tspan x="${x + w / 2}" dy="${index === 0 ? 0 : lineHeight}">${esc(value)}</tspan>`)
    .join("")}</text>`;
}

function box(x, y, w, h, title, body = [], theme = "gray", options = {}) {
  const [fill, stroke] = themes[theme];
  const dash = options.dashed ? ' stroke-dasharray="10 7"' : "";
  const titleLines = Array.isArray(title) ? title : [title];
  const titleTop = options.titleTop ?? y + 36;
  const titleLineHeight = options.titleLineHeight || 27;
  const titleFont = options.titleFont || 24;
  const bodyTop = options.bodyTop || titleTop + titleLineHeight * titleLines.length + (options.bodyGap ?? 13);
  const bodyClass = options.bodyClass || "body";
  const bodyLineHeight = options.bodyLineHeight || 28;
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${options.strokeWidth || 2.4}"${dash}/>`;
  out += titleBlock(x, titleTop, w, titleLines, { fontSize: titleFont, lineHeight: titleLineHeight });
  if (body.length) {
    out += textLines(x + w / 2, bodyTop, body, {
      cls: bodyClass,
      lineHeight: bodyLineHeight,
      fill: options.bodyFill,
    });
  }
  return out;
}

function stage(x, y, w, h, title, color, fill) {
  return `<g>
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#FFFFFF" stroke="${color}" stroke-width="2.4"/>
    <path d="M${x + 10},${y} H${x + w - 10} Q${x + w},${y} ${x + w},${y + 10} V${y + 58} H${x} V${y + 10} Q${x},${y} ${x + 10},${y} Z" fill="${fill}"/>
    <line x1="${x}" y1="${y + 58}" x2="${x + w}" y2="${y + 58}" stroke="${color}" stroke-width="2"/>
    <text x="${x + w / 2}" y="${y + 39}" text-anchor="middle" class="stage-title">${esc(title)}</text>
  </g>`;
}

function smallTag(x, y, w, h, label, theme = "gray", options = {}) {
  const [fill, stroke] = themes[theme];
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="1.8"${options.dashed ? ' stroke-dasharray="7 5"' : ""}/>
    <text x="${x + w / 2}" y="${y + h / 2 + 7}" text-anchor="middle" class="tag">${esc(label)}</text>`;
}

function poly(points, options = {}) {
  const color = options.color || C.line;
  const marker = options.marker === false ? "" : ` marker-end="url(#${options.marker || "arrow-dark"})"`;
  const dash = options.dashed ? ' stroke-dasharray="10 7"' : "";
  const width = options.width || 2.6;
  return `<polyline points="${points.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="square" stroke-linejoin="round"${dash}${marker}/>`;
}

function note(x, y, lines, color = C.muted, options = {}) {
  const values = Array.isArray(lines) ? lines : [lines];
  const width = options.width || 260;
  const h = 20 + values.length * (options.lineHeight || 24);
  return `<rect x="${x}" y="${y}" width="${width}" height="${h}" rx="5" fill="#FFFFFF" stroke="${color}" stroke-width="1.5"${options.dashed ? ' stroke-dasharray="7 5"' : ""}/>
    ${textLines(x + width / 2, y + 27, values, { cls: "small", lineHeight: options.lineHeight || 24, fill: color })}`;
}

function lossBox(x, y, w, label) {
  const lines = Array.isArray(label) ? label : [label];
  return `<rect x="${x}" y="${y}" width="${w}" height="82" rx="6" fill="#FFFFFF" stroke="${C.gray}" stroke-width="1.7"/>
    ${textLines(x + w / 2, y + (lines.length === 1 ? 48 : 34), lines, { cls: "loss", lineHeight: 26 })}`;
}

function buildSvg() {
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">
  <title id="title">船舶轨迹预测模型总体框架</title>
  <desc id="desc">面向人工智能与交通运输论文的五阶段船舶轨迹预测模型框架，包含多源输入、双分支编码、层级航路意图、航路条件生成、候选选择和联合训练目标。</desc>
  <defs>
    <marker id="arrow-dark" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.line}"/></marker>
    <marker id="arrow-blue" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.blue}"/></marker>
    <marker id="arrow-teal" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.teal}"/></marker>
    <marker id="arrow-green" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.greenDark}"/></marker>
    <marker id="arrow-orange" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.orangeDark}"/></marker>
    <marker id="arrow-red" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.red}"/></marker>
    <style>
      text { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; fill: ${C.ink}; letter-spacing: 0; }
      .figure-title { font-size: 38px; font-weight: 700; }
      .stage-title { font-size: 27px; font-weight: 700; }
      .body { font-size: 20px; font-weight: 400; }
      .small { font-size: 17px; font-weight: 400; }
      .tag { font-size: 19px; font-weight: 600; }
      .loss { font-size: 18px; font-weight: 600; }
      .group-title { font-size: 21px; font-weight: 700; }
      .tensor { font-size: 17px; font-weight: 500; fill: ${C.muted}; }
    </style>
  </defs>
  <rect x="0" y="0" width="${W}" height="${H}" fill="${C.paper}"/>
  <text x="1600" y="56" text-anchor="middle" class="figure-title">船舶轨迹预测模型总体框架</text>
  <g aria-label="legend">
    <line x1="2485" y1="49" x2="2555" y2="49" stroke="${C.line}" stroke-width="2.6" marker-end="url(#arrow-dark)"/>
    <text x="2570" y="56" class="small">推理数据流</text>
    <line x1="2760" y1="49" x2="2830" y2="49" stroke="${C.red}" stroke-width="2.6" stroke-dasharray="10 7" marker-end="url(#arrow-red)"/>
    <text x="2845" y="56" class="small" fill="${C.red}">训练专用监督</text>
  </g>`;

  // Five stage containers.
  s += stage(40, 92, 420, 1270, "第一阶段  多源输入", C.blue, C.blueFill);
  s += stage(480, 92, 560, 1270, "第二阶段  双分支特征编码", C.blue, C.blueFill);
  s += stage(1060, 92, 720, 1270, "第三阶段  层级航路意图推断", C.greenDark, C.greenFill);
  s += stage(1800, 92, 680, 1270, "第四阶段  航路条件轨迹生成", C.orangeDark, C.orangeFill);
  s += stage(2500, 92, 660, 1270, "第五阶段  候选选择与输出", C.gray, C.grayFill);

  // Stage 1: multi-source inputs.
  s += box(70, 210, 360, 170, "历史AIS轨迹", ["3小时 · 13个观测点", "经纬度、SOG、COG", "及对应变化量"], "blue", { bodyTop: 292, bodyLineHeight: 29 });
  s += box(70, 470, 360, 150, "航次语义信息", ["船型、吃水、Destination", "航行状态"], "teal", { bodyTop: 550, bodyLineHeight: 29 });
  s += box(70, 735, 360, 205, "训练专用监督输入", ["主航路标签 · 子航路标签", "真实未来轨迹", "仅训练可用 · 推理阶段不可用"], "red", { dashed: true, bodyTop: 817, bodyLineHeight: 32, bodyFill: C.red });

  // Stage 2: dual-branch motion and semantic encoding.
  s += box(510, 205, 500, 180, "TCN局部行为编码器", ["短期运动变化 · 转向趋势", "局部意图序列"], "blue", { bodyTop: 288, bodyLineHeight: 31 });
  s += smallTag(832, 334, 150, 38, "意图摘要", "blue");
  s += box(510, 470, 210, 108, "输入映射", ["+ 位置编码"], "blue", { titleFont: 22, bodyTop: 544, bodyLineHeight: 25 });
  s += box(755, 450, 255, 150, ["Transformer", "轨迹编码器"], ["全局时序运动特征"], "blue", { titleTop: 485, titleFont: 22, titleLineHeight: 25, bodyTop: 570, bodyClass: "small" });
  s += box(510, 690, 500, 190, "冻结Qwen3-1.7B语义教师", ["离线预编码 → 语义向量", "Qwen不直接生成轨迹", "Qwen不参与在线坐标回归"], "teal", { bodyTop: 770, bodyLineHeight: 29, bodyFill: C.teal });
  s += box(510, 960, 210, 112, "语义投影MLP", ["维度映射"], "teal", { titleFont: 22, bodyTop: 1033, bodyClass: "small" });
  s += box(755, 930, 255, 165, "门控语义融合", ["语义向量 × 意图摘要", "输出融合特征"], "teal", { titleFont: 23, bodyTop: 1012, bodyLineHeight: 29 });

  // Stage 1 to stage 2 arrows.
  s += poly([[430, 295], [470, 295], [470, 275], [510, 275]], { color: C.blue, marker: "arrow-blue" });
  s += poly([[450, 295], [450, 524], [510, 524]], { color: C.blue, marker: "arrow-blue" });
  s += poly([[430, 545], [470, 545], [470, 785], [510, 785]], { color: C.teal, marker: "arrow-teal" });
  s += poly([[720, 524], [755, 524]], { color: C.blue, marker: "arrow-blue" });
  s += poly([[615, 880], [615, 960]], { color: C.teal, marker: "arrow-teal" });
  s += poly([[720, 1016], [755, 1016]], { color: C.teal, marker: "arrow-teal" });
  s += poly([[1010, 292], [1024, 292], [1024, 995], [1010, 995]], { color: C.teal, marker: "arrow-teal" });
  s += textLines(998, 914, ["TCN意图摘要"], { cls: "tensor", anchor: "end", fill: C.teal });

  // Stage 3: hierarchical route intention.
  s += box(1120, 240, 600, 82, "融合特征", ["运动表征 + 门控语义"], "green", { titleTop: 274, titleFont: 25, bodyTop: 305, bodyClass: "small" });
  s += box(1090, 375, 180, 108, ["训练集主航路", "原型先验"], [], "green", { titleTop: 413, titleFont: 20, titleLineHeight: 24 });
  s += box(1300, 355, 270, 155, "大类航路意图头", ["层级大类分类"], "green", { bodyTop: 441 });
  s += box(1595, 375, 155, 108, ["大类可判别性", "门控"], [], "green", { titleTop: 413, titleFont: 19, titleLineHeight: 24 });
  s += box(1300, 535, 270, 78, "P(route)", ["OA · OB1 · OB2 · OC"], "green", { titleTop: 566, titleFont: 22, bodyTop: 598, bodyClass: "small" });
  s += box(1595, 535, 155, 90, ["置信度门控", "层级约束"], [], "green", { titleTop: 570, titleFont: 18, titleLineHeight: 23 });
  s += box(1090, 690, 180, 108, ["训练集子航路", "原型先验"], [], "green", { titleTop: 728, titleFont: 20, titleLineHeight: 24 });
  s += box(1090, 830, 180, 118, ["未来增强", "意图原型"], ["训练专用"], "red", { dashed: true, titleTop: 867, titleFont: 20, titleLineHeight: 24, bodyTop: 928, bodyClass: "small", bodyFill: C.red });
  s += box(1300, 680, 270, 185, "子航路意图头", ["原型先验 + 未来增强", "层级约束 + 可判别门控"], "green", { bodyTop: 770, bodyLineHeight: 30 });
  s += box(1595, 700, 155, 108, ["子航路可判别性", "门控"], [], "green", { titleTop: 738, titleFont: 18, titleLineHeight: 24 });
  s += box(1300, 895, 270, 78, "P(subroute)", ["各子航路概率"], "green", { titleTop: 926, titleFont: 21, bodyTop: 958, bodyClass: "small" });
  s += box(1170, 1010, 500, 135, "置信度感知路由", ["高置信度：确定单一航路", "低置信度：保留Top-K候选"], "green", { bodyTop: 1091, bodyLineHeight: 29 });
  s += box(1100, 1190, 250, 96, "Route Embedding", ["主航路嵌入"], "green", { titleTop: 1226, titleFont: 21, bodyTop: 1265, bodyClass: "small" });
  s += box(1450, 1190, 280, 96, "Subroute Embedding", ["子航路嵌入"], "green", { titleTop: 1226, titleFont: 20, bodyTop: 1265, bodyClass: "small" });

  // Stage 2 to stage 3 fusion path.
  s += poly([[1010, 1012], [1030, 1012], [1030, 281], [1120, 281]], { color: C.greenDark, marker: "arrow-green", width: 3 });
  s += poly([[1420, 322], [1420, 355]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1270, 429], [1300, 429]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1595, 429], [1570, 429]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1435, 510], [1435, 535]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1570, 574], [1595, 574]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1672, 625], [1672, 658], [1485, 658], [1485, 680]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1720, 281], [1762, 281], [1762, 772], [1570, 772]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1270, 744], [1300, 744]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1270, 889], [1284, 889], [1284, 820], [1300, 820]], { color: C.red, marker: "arrow-red", dashed: true });
  s += poly([[1595, 754], [1570, 754]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1435, 865], [1435, 895]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1435, 973], [1435, 1010]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1325, 613], [1282, 613], [1282, 1170], [1225, 1170], [1225, 1190]], { color: C.greenDark, marker: "arrow-green" });
  s += poly([[1530, 1145], [1590, 1145], [1590, 1190]], { color: C.greenDark, marker: "arrow-green" });

  // Embedding feedback into TCN. Two parallel orthogonal feedback channels stay in the stage gutter.
  s += poly([[1100, 1238], [1052, 1238], [1052, 255], [1010, 255]], { color: C.greenDark, marker: "arrow-green", width: 2.3 });
  s += poly([[1450, 1258], [1043, 1258], [1043, 330], [1010, 330]], { color: C.greenDark, marker: "arrow-green", width: 2.3 });
  s += textLines(1068, 1313, ["航路嵌入反馈至TCN意图序列"], { cls: "tensor", anchor: "start", fill: C.greenDark });

  // Global Transformer and route-corrected TCN feature highways to stage 4.
  s += poly([[1010, 510], [1037, 510], [1037, 177], [1830, 177], [1830, 287]], { color: C.blue, marker: "arrow-blue", width: 3 });
  s += textLines(1370, 171, ["Transformer全局轨迹特征"], { cls: "tensor", fill: C.blue });
  s += poly([[1010, 235], [1034, 235], [1034, 194], [1850, 194], [1850, 417]], { color: C.blue, marker: "arrow-blue", width: 3 });
  s += textLines(1455, 218, ["航路嵌入修正的TCN局部意图特征"], { cls: "tensor", fill: C.blue });

  // Stage 4: route-conditioned trajectory generation.
  s += box(1830, 235, 620, 102, "Transformer全局轨迹特征", ["全局时序运动表征"], "blue", { titleTop: 271, bodyTop: 313, bodyClass: "small" });
  s += box(1830, 365, 620, 102, "航路嵌入修正的TCN局部意图特征", ["局部转向与意图序列"], "blue", { titleTop: 401, titleFont: 22, bodyTop: 443, bodyClass: "small" });
  s += box(1960, 525, 360, 118, "特征融合块 Fusion Block", ["全局运动 × 局部意图"], "orange", { titleTop: 563, titleFont: 23, bodyTop: 613 });
  s += box(1960, 700, 360, 125, "共享轨迹解码器", ["学习公共运动规律", "输出共享预测残差"], "orange", { bodyTop: 781, bodyLineHeight: 28 });
  s += box(1915, 875, 450, 170, "子航路残差专家库", ["依据P(subroute)加权", "修正分支特有转向", "与横向偏移"], "orange", { bodyTop: 958, bodyLineHeight: 28 });
  s += box(1915, 1090, 450, 118, "多航路条件候选轨迹", ["每个候选航路生成一条轨迹"], "orange", { bodyTop: 1173 });
  s += box(1835, 1240, 245, 102, ["线性运动", "外推基线"], [], "gray", { titleTop: 1276, titleFont: 20, titleLineHeight: 23 });
  s += `<circle cx="2135" cy="1291" r="25" fill="#FFFFFF" stroke="${C.orangeDark}" stroke-width="2.8"/><text x="2135" y="1301" text-anchor="middle" font-size="31" font-weight="700" fill="${C.orangeDark}">+</text>`;
  s += box(2190, 1235, 260, 112, ["未来绝对轨迹", "候选集"], [], "orange", { titleTop: 1274, titleFont: 21, titleLineHeight: 25 });
  s += textLines(2135, 1342, ["神经残差 + 线性基线"], { cls: "tensor", fill: C.orangeDark });

  s += poly([[2050, 337], [2050, 495], [2075, 495], [2075, 525]], { color: C.orangeDark, marker: "arrow-orange" });
  s += poly([[2250, 467], [2250, 495], [2205, 495], [2205, 525]], { color: C.orangeDark, marker: "arrow-orange" });
  s += poly([[2140, 643], [2140, 700]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[2140, 825], [2140, 875]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[1570, 935], [1790, 935], [1790, 952], [1915, 952]], { color: C.greenDark, marker: "arrow-green" });
  s += textLines(1710, 924, ["P(subroute)"], { cls: "tensor", fill: C.greenDark });
  s += poly([[2140, 1045], [2140, 1090]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[2140, 1208], [2140, 1266]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[2080, 1291], [2110, 1291]], { color: C.gray, marker: "arrow-dark" });
  s += poly([[2160, 1291], [2190, 1291]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });

  // Stage 5: candidate selection and output.
  s += `<rect x="2530" y="210" width="600" height="340" rx="8" fill="${C.grayFill}" stroke="${C.gray}" stroke-width="2.2"/>
    <text x="2830" y="247" text-anchor="middle" class="group-title">选择器输入特征</text>`;
  s += smallTag(2555, 275, 265, 62, "主航路概率", "green");
  s += smallTag(2840, 275, 265, 62, "子航路概率", "green");
  s += smallTag(2555, 362, 265, 62, "历史连续性", "gray");
  s += smallTag(2840, 362, 265, 62, "轨迹平滑性", "gray");
  s += smallTag(2555, 449, 265, 62, "航路原型距离", "green");
  s += smallTag(2840, 449, 265, 62, "候选嵌入", "orange");
  s += box(2580, 620, 500, 155, "学习型候选选择器", ["融合候选质量与航路先验", "输出候选排序得分"], "orange", { bodyTop: 704, bodyLineHeight: 29 });
  s += box(2680, 855, 300, 105, "验证集阈值校准", ["校准置信度与切换阈值"], "gray", { titleTop: 892, titleFont: 22, bodyTop: 934, bodyClass: "small" });
  s += box(2580, 1040, 500, 235, "唯一Top-1轨迹", ["未来3小时 · 12个预测点", "经纬度、SOG、COG"], "orange", { titleTop: 1092, titleFont: 29, bodyTop: 1163, bodyLineHeight: 36 });
  s += poly([[2830, 550], [2830, 620]], { color: C.gray, marker: "arrow-dark" });
  s += poly([[2450, 1291], [2488, 1291], [2488, 698], [2580, 698]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[2830, 775], [2830, 855]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });
  s += poly([[2830, 960], [2830, 1040]], { color: C.orangeDark, marker: "arrow-orange", width: 3 });

  // Training-only input bus above the loss band.
  s += poly([[250, 940], [250, 1390], [1582, 1390]], { color: C.red, marker: false, dashed: true, width: 2.5 });
  s += textLines(475, 1380, ["主/子航路标签与真实未来轨迹"], { cls: "tensor", fill: C.red });
  s += poly([[1582, 1390], [1582, 438], [1570, 438]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += poly([[1582, 805], [1570, 805]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += poly([[1100, 1390], [1078, 1390], [1078, 889], [1090, 889]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });

  // Bottom training objective band.
  s += `<rect x="40" y="1420" width="3120" height="335" rx="10" fill="${C.grayBand}" stroke="${C.gray}" stroke-width="2.2"/>
    <rect x="70" y="1402" width="250" height="42" rx="6" fill="#FFFFFF" stroke="${C.gray}" stroke-width="1.8"/>
    <text x="195" y="1431" text-anchor="middle" class="stage-title">联合训练目标</text>`;

  // Loss groups.
  s += `<rect x="65" y="1470" width="1085" height="235" rx="8" fill="#FFFFFF" stroke="${C.border}" stroke-width="1.7"/>
    <text x="607" y="1506" text-anchor="middle" class="group-title">轨迹生成损失（自然分布轨迹回归主流）</text>`;
  s += lossBox(83, 1540, 198, "轨迹回归损失");
  s += lossBox(295, 1540, 198, "地理距离损失");
  s += lossBox(507, 1540, 198, ["FDE终点", "损失"]);
  s += lossBox(719, 1540, 198, "轨迹平滑损失");
  s += lossBox(931, 1540, 198, ["循环COG", "损失"]);
  s += textLines(607, 1670, ["监督共享解码、残差专家与绝对轨迹恢复"], { cls: "small", fill: C.gray });

  s += `<rect x="1170" y="1470" width="1065" height="235" rx="8" fill="#FFFFFF" stroke="${C.border}" stroke-width="1.7"/>
    <text x="1702" y="1506" text-anchor="middle" class="group-title">层级意图推断损失</text>`;
  s += lossBox(1188, 1540, 192, ["主航路分类", "损失"]);
  s += lossBox(1397, 1540, 192, ["子航路", "Focal Loss"]);
  s += lossBox(1606, 1540, 192, ["可判别性", "损失"]);
  s += lossBox(1815, 1540, 192, ["监督对比", "损失"]);
  s += lossBox(2024, 1540, 192, ["未来意图对齐", "损失"]);
  s += textLines(1702, 1670, ["监督大类/子航路分类、门控与意图表示"], { cls: "small", fill: C.gray });

  s += `<rect x="2255" y="1470" width="300" height="235" rx="8" fill="#FFFFFF" stroke="${C.border}" stroke-width="1.7"/>
    <text x="2405" y="1506" text-anchor="middle" class="group-title">候选选择损失</text>`;
  s += lossBox(2300, 1550, 210, ["候选排序", "损失"]);
  s += textLines(2405, 1670, ["监督Top-1选择"], { cls: "small", fill: C.gray });

  s += box(2580, 1470, 550, 235, "小类均衡辅助意图流", ["仅增强航路分类与意图表示", "不改变自然分布轨迹回归主流"], "gray", { titleTop: 1512, titleFont: 24, bodyTop: 1580, bodyLineHeight: 36 });

  // Red dashed links from objectives to the corresponding modules.
  s += poly([[607, 1470], [607, 1404], [1818, 1404], [1818, 762], [1960, 762]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += poly([[1702, 1470], [1702, 1398], [1588, 1398]], { color: C.red, marker: false, dashed: true, width: 2.5 });
  s += poly([[1588, 1140], [1670, 1140]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += poly([[2405, 1470], [2405, 1382], [2496, 1382], [2496, 735], [2580, 735]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += poly([[2855, 1470], [2855, 1374], [1774, 1374], [1774, 798], [1570, 798]], { color: C.red, marker: "arrow-red", dashed: true, width: 2.5 });
  s += textLines(2935, 1738, ["红色虚线：训练专用；推理阶段移除"], { cls: "small", anchor: "end", fill: C.red });

  return s + "</svg>\n";
}

async function main() {
  const svg = buildSvg();
  fs.writeFileSync(SVG_PATH, svg, "utf8");
  console.log(SVG_PATH);
  if (sharp) {
    await sharp(Buffer.from(svg), { density: 150 })
      .resize(W, H, { fit: "fill" })
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .withMetadata({ density: 300 })
      .toFile(PNG_PATH);
    console.log(PNG_PATH);
  } else {
    console.warn("sharp unavailable; SVG generated, PNG export skipped");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
