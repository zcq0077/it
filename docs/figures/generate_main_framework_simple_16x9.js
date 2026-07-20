const fs = require("fs");
const path = require("path");

const W = 3200;
const H = 1800;
const SVG_PATH = path.join(__dirname, "fig1-paper-main-framework-16x9.svg");

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
  green: "#2E7D32",
  greenFill: "#EDF7ED",
  orange: "#B66F00",
  orangeFill: "#FFF4DE",
  red: "#8B1E2D",
  redFill: "#FCECEE",
  gray: "#6B7280",
  grayFill: "#F3F4F6",
  band: "#F7F8FA",
};

const themes = {
  blue: [C.blueFill, C.blue],
  teal: [C.tealFill, C.teal],
  green: [C.greenFill, C.green],
  orange: [C.orangeFill, C.orange],
  red: [C.redFill, C.red],
  gray: [C.grayFill, C.gray],
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
  const lineHeight = options.lineHeight || 30;
  const fill = options.fill ? ` fill="${options.fill}"` : "";
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}"${fill}>${list
    .map((value, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${esc(value)}</tspan>`)
    .join("")}</text>`;
}

function stage(x, y, w, h, title, stroke, fill) {
  return `<g>
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${fill}" fill-opacity="0.12" stroke="${stroke}" stroke-width="2.2"/>
    <path d="M${x + 10},${y} H${x + w - 10} Q${x + w},${y} ${x + w},${y + 10} V${y + 60} H${x} V${y + 10} Q${x},${y} ${x + 10},${y} Z" fill="${fill}"/>
    <line x1="${x}" y1="${y + 60}" x2="${x + w}" y2="${y + 60}" stroke="${stroke}" stroke-width="2"/>
    <text x="${x + w / 2}" y="${y + 41}" text-anchor="middle" class="stage-title">${esc(title)}</text>
  </g>`;
}

function box(x, y, w, h, title, body = [], theme = "gray", options = {}) {
  const [fill, stroke] = themes[theme];
  const dashed = options.dashed ? ' stroke-dasharray="10 7"' : "";
  const titleLines = Array.isArray(title) ? title : [title];
  const titleY = options.titleY ?? y + 42;
  const titleSize = options.titleSize || 26;
  const titleLineHeight = options.titleLineHeight || 29;
  const bodyY = options.bodyY ?? titleY + titleLines.length * titleLineHeight + 18;
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${options.strokeWidth || 2.4}"${dashed}/>`;
  out += `<text x="${x + w / 2}" y="${titleY}" text-anchor="middle" font-size="${titleSize}" font-weight="700">${titleLines
    .map((line, index) => `<tspan x="${x + w / 2}" dy="${index === 0 ? 0 : titleLineHeight}">${esc(line)}</tspan>`)
    .join("")}</text>`;
  if (body.length) {
    out += textLines(x + w / 2, bodyY, body, {
      cls: options.bodyClass || "body",
      lineHeight: options.bodyLineHeight || 30,
      fill: options.bodyFill,
    });
  }
  return out;
}

function tag(x, y, w, h, label, theme = "gray") {
  const [fill, stroke] = themes[theme];
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="1.8"/>
    <text x="${x + w / 2}" y="${y + h / 2 + 7}" text-anchor="middle" class="tag">${esc(label)}</text>`;
}

function arrow(points, options = {}) {
  const color = options.color || C.line;
  const marker = options.marker === false ? "" : ` marker-end="url(#${options.marker || "arrow-dark"})"`;
  return `<polyline points="${points.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="${color}" stroke-width="${options.width || 2.8}" stroke-linecap="square" stroke-linejoin="round"${options.dashed ? ' stroke-dasharray="10 7"' : ""}${marker}/>`;
}

function build() {
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">
  <title id="title">面向层级航路意图的船舶轨迹预测框架</title>
  <desc id="desc">五阶段船舶轨迹预测推理框架，突出双分支特征编码、层级航路意图、条件轨迹生成和候选选择。</desc>
  <defs>
    <marker id="arrow-dark" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.line}"/></marker>
    <marker id="arrow-blue" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.blue}"/></marker>
    <marker id="arrow-teal" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.teal}"/></marker>
    <marker id="arrow-green" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.green}"/></marker>
    <marker id="arrow-orange" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.orange}"/></marker>
    <marker id="arrow-red" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L11,4.5 L0,9 Z" fill="${C.red}"/></marker>
    <style>
      text { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; fill: ${C.ink}; letter-spacing: 0; }
      .figure-title { font-size: 40px; font-weight: 700; }
      .stage-title { font-size: 28px; font-weight: 700; }
      .body { font-size: 21px; font-weight: 400; }
      .small { font-size: 18px; font-weight: 400; }
      .tag { font-size: 20px; font-weight: 600; }
      .group-title { font-size: 22px; font-weight: 700; }
      .tensor { font-size: 18px; font-weight: 500; fill: ${C.muted}; }
    </style>
  </defs>
  <rect x="0" y="0" width="${W}" height="${H}" fill="${C.paper}"/>
  <text x="1600" y="60" text-anchor="middle" class="figure-title">面向层级航路意图的船舶轨迹预测框架</text>
  <g aria-label="module legend">
    <rect x="58" y="34" width="24" height="18" rx="3" fill="${C.blueFill}" stroke="${C.blue}" stroke-width="1.8"/>
    <text x="94" y="51" class="small">运动编码</text>
    <rect x="220" y="34" width="24" height="18" rx="3" fill="${C.tealFill}" stroke="${C.teal}" stroke-width="1.8"/>
    <text x="256" y="51" class="small">语义信息</text>
    <rect x="382" y="34" width="24" height="18" rx="3" fill="${C.greenFill}" stroke="${C.green}" stroke-width="1.8"/>
    <text x="418" y="51" class="small">航路意图</text>
    <rect x="544" y="34" width="24" height="18" rx="3" fill="${C.orangeFill}" stroke="${C.orange}" stroke-width="1.8"/>
    <text x="580" y="51" class="small">轨迹生成</text>
    <rect x="706" y="34" width="24" height="18" rx="3" fill="${C.grayFill}" stroke="${C.gray}" stroke-width="1.8"/>
    <text x="742" y="51" class="small">选择输出</text>
  </g>`;

  // Stage containers.
  s += stage(40, 100, 420, 1560, "第一阶段  多源输入", C.blue, C.blueFill);
  s += stage(480, 100, 560, 1560, "第二阶段  运动特征编码", C.blue, C.blueFill);
  s += stage(1060, 100, 720, 1560, "第三阶段  航路意图推断", C.green, C.greenFill);
  s += stage(1800, 100, 680, 1560, "第四阶段  条件轨迹生成", C.orange, C.orangeFill);
  s += stage(2500, 100, 660, 1560, "第五阶段  候选选择与输出", C.gray, C.grayFill);

  // Stage 1.
  s += box(70, 380, 360, 180, "历史AIS轨迹", ["3小时 · 13个观测点", "经纬度、SOG、COG及变化量"], "blue", { bodyY: 473, bodyLineHeight: 32 });
  s += box(70, 1010, 360, 170, "航次语义信息", ["船型 · 吃水 · Destination", "航行状态"], "teal", { bodyY: 1101, bodyLineHeight: 32 });

  // Stage 2.
  s += `<rect x="510" y="270" width="500" height="430" rx="8" fill="${C.blueFill}" stroke="${C.blue}" stroke-width="2.6"/>
    <text x="760" y="319" text-anchor="middle" font-size="28" font-weight="700">全局/局部双分支运动编码器</text>
    <line x1="545" y1="345" x2="975" y2="345" stroke="${C.blue}" stroke-width="1.5" opacity="0.65"/>
    <rect x="550" y="375" width="420" height="105" rx="6" fill="#FFFFFF" fill-opacity="0.82" stroke="${C.blue}" stroke-width="1.8"/>
    <text x="760" y="416" text-anchor="middle" font-size="24" font-weight="700">Transformer全局轨迹编码</text>
    <text x="760" y="453" text-anchor="middle" class="small">输入映射 + 位置编码</text>
    <rect x="550" y="535" width="420" height="105" rx="6" fill="#FFFFFF" fill-opacity="0.82" stroke="${C.blue}" stroke-width="1.8"/>
    <text x="760" y="576" text-anchor="middle" font-size="24" font-weight="700">TCN局部行为编码</text>
    <text x="760" y="613" text-anchor="middle" class="small">短期变化 · 转向趋势</text>`;
  s += arrow([[510, 488], [530, 488], [530, 428], [550, 428]], { color: C.blue, marker: "arrow-blue", width: 2.4 });
  s += arrow([[530, 488], [530, 588], [550, 588]], { color: C.blue, marker: "arrow-blue", width: 2.4 });
  s += box(510, 1010, 500, 220, "冻结Qwen3-1.7B语义教师", ["离线预编码 → 语义投影MLP", "门控融合TCN意图摘要", "不直接生成轨迹"], "teal", { bodyY: 1104, bodyLineHeight: 33, bodyFill: C.teal });

  // Stage 3 is deliberately summarized in the main figure; details belong in a subfigure.
  s += box(1180, 650, 480, 250, "层级航路意图模块", [
    "主航路 / 子航路联合推断",
    "航路先验与置信度路由",
    "输出航路条件嵌入",
  ], "green", { titleY: 704, titleSize: 29, bodyY: 768, bodyLineHeight: 38 });
  s += textLines(1420, 972, ["层级结构、门控与Top-K细节见意图子图"], { cls: "tensor", fill: C.green });

  // Stage 4.
  s += box(1840, 270, 600, 175, "特征融合块 Fusion Block", ["Transformer全局特征", "+ 航路修正的TCN局部意图"], "orange", { bodyY: 361, bodyLineHeight: 32 });
  s += box(1840, 680, 600, 220, ["共享轨迹解码器", "+ 子航路残差专家库"], ["公共运动规律 + 分支转向修正", "依据P(subroute)组合专家残差"], "orange", { titleY: 727, titleLineHeight: 33, bodyY: 829, bodyLineHeight: 32 });
  s += box(1840, 1100, 600, 205, "多航路条件候选轨迹", ["神经网络预测残差 + 线性运动基线", "恢复未来绝对轨迹候选集"], "orange", { bodyY: 1195, bodyLineHeight: 34 });

  // Stage 5.
  s += box(2550, 430, 560, 245, "学习型候选选择器", ["航路概率 · 连续性 · 平滑性", "原型距离 · 候选嵌入", "候选排序得分"], "orange", { bodyY: 528, bodyLineHeight: 34 });
  s += box(2640, 845, 380, 125, "验证集阈值校准", ["置信度与切换阈值"], "gray", { bodyY: 929, bodyClass: "small" });
  s += box(2550, 1140, 560, 225, "唯一Top-1轨迹", ["未来3小时 · 12个预测点", "经纬度、SOG、COG"], "orange", { titleY: 1196, titleSize: 31, bodyY: 1280, bodyLineHeight: 38, strokeWidth: 3 });

  // Main inference arrows: all segments are orthogonal.
  s += arrow([[430, 470], [470, 470], [470, 488], [510, 488]], { color: C.blue, marker: "arrow-blue" });
  s += arrow([[430, 1095], [470, 1095], [470, 1120], [510, 1120]], { color: C.teal, marker: "arrow-teal" });
  s += arrow([[970, 588], [1045, 588], [1045, 710], [1180, 710]], { color: C.blue, marker: "arrow-blue", width: 3 });
  s += arrow([[1010, 1120], [1080, 1120], [1080, 840], [1180, 840]], { color: C.teal, marker: "arrow-teal", width: 3 });

  // Transformer bypass and route-conditioned local feature enter the fusion block.
  s += arrow([[970, 428], [1035, 428], [1035, 190], [2010, 190], [2010, 270]], { color: C.blue, marker: "arrow-blue", width: 3 });
  s += textLines(1480, 184, ["全局轨迹特征"], { cls: "tensor", fill: C.blue });
  s += arrow([[1660, 775], [1785, 775], [1785, 390], [1840, 390]], { color: C.green, marker: "arrow-green", width: 3 });
  s += textLines(1770, 755, ["航路条件嵌入"], { cls: "tensor", anchor: "end", fill: C.green });
  s += arrow([[2140, 445], [2140, 680]], { color: C.orange, marker: "arrow-orange", width: 3 });
  s += arrow([[2140, 900], [2140, 1100]], { color: C.orange, marker: "arrow-orange", width: 3 });
  s += arrow([[2440, 1202], [2485, 1202], [2485, 550], [2550, 550]], { color: C.orange, marker: "arrow-orange", width: 3 });
  s += textLines(2470, 1180, ["候选轨迹集"], { cls: "tensor", anchor: "end", fill: C.orange });
  s += arrow([[2830, 675], [2830, 845]], { color: C.orange, marker: "arrow-orange", width: 3 });
  s += arrow([[2830, 970], [2830, 1140]], { color: C.orange, marker: "arrow-orange", width: 3 });

  return s + "</svg>\n";
}

fs.writeFileSync(SVG_PATH, build(), "utf8");
console.log(SVG_PATH);
