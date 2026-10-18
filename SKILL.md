---
name: bitmap-to-vector-svg
description: 位图/插画/贴纸/图标 → 真矢量 SVG（path + fill，非栅格 image 嵌入）。当用户给出一张彩色插画、贴纸、动漫图、图标或线稿，要求"转矢量""矢量化""导出可编辑 SVG""线稿上色""SVG 分层填色"时使用。流程为三阶段：生图工具抽线稿（提示词必须声明线条归一化）→ 线稿矢量化（骨架化 + 最少基元拟合）→ 颜色填充（B+C 分层合成）。也适用于"已有线稿只做矢量化"或"已有线稿只做上色"的半程任务。
agent_created: true
---

# 位图 → 真矢量 SVG

产出的是**可编辑的 path/fill 结构**（分层色块 + 单 path 线稿），不是把位图塞进 `<image>`。

成品 SVG 结构（自上而下图层顺序）：

```svg
<rect .../>                    白底
<g id="fills-base"> ... </g>   底层补齐（C：泛洪分区 + 边缘吸附，负责背景/开环）
<g id="fills-main"> ... </g>   主基底（B：骨架环面遍历，闭合区色块天生贴线）
<path id="lineart" .../>       线稿（单 path，统一线宽）
```

## 何时使用

- 彩色插画 / 贴纸 / 动漫立绘 / 图标 → 可编辑 SVG
- 已有线稿 → 矢量线稿；已有线稿 + 原彩图 → 分层上色 SVG

## 何时不用（先说清边界）

- **真实照片 / 连续色调扫描件**：当前管线做不到照片级渐变，需要另开渐变 mesh 路线
- 截图 / UI / 文字：直接重绘或 OCR 更合适

## 三阶段总览

| 阶段 | 输入 | 输出 | 工具 |
|---|---|---|---|
| 1. 抽线稿 | 彩色位图 | 干净二值线稿 png | **生图工具（ImageGen 图生图）**，不是本地 CV |
| 2. 矢量化 | 线稿 png | `<name>_lineart.svg` + `<name>_normmask.png` | `scripts/pipeline.py --stage lineart` |
| 3. 上色 | 掩膜 + 线稿 svg + 原彩图 | `<name>_colored.svg`（正式成品） | `scripts/pipeline.py --stage fill` |

工作目录约定：素材与产物都放在 `output/`（默认 = 当前工作目录/output，可用 `--outdir` 改）。
标准文件名：线稿 `output/lineart_src.png`、彩图 `output/original.jpg`（脚本自动探测，也可用 `--src-lineart/--src-color` 显式指定）。
`<name>` 由 `--name` 指定（默认 `out`），用于区分同一目录下的不同任务。

---

## 阶段 1：位图 → 干净线稿（关键，别用本地 CV）

**本地古典 CV 抽取（黑帽变换 / XDoG / 照度平坦化）在插画上必然被阴影、噪点、抗锯齿污染，完整度差一个档次。正解是用生图工具把原图重绘成线稿。**

1. 用 ImageGen 图生图，传入原图，`input_fidelity=high`，尺寸长边 1024。
2. **提示词必须声明"线条归一化"**——这一步决定后续矢量化能否成功：
   - 所有线条**统一粗细**（同一恒定线宽，无粗细变化、无笔锋、无压感）
   - **纯黑白二值**（只有纯黑线条 + 纯白背景；禁止灰阶、抗锯齿灰边、阴影、渐变、涂色）
   - 轮廓**闭合**、结构完整、不丢细节、不新增元素
   - 白底、无纹理、无网点、无噪点
3. 完整提示词模板见 `references/lineart_prompt.md`（中英两版，直接取用）。
4. 生成结果另存为 `output/lineart_src.png`。

> 为什么必须归一化：后续管线会把线条统一重渲染成固定线宽再骨架化。原线稿粗细不均时，骨架化会在粗线处产生分叉/毛刺，拟合基元数暴涨。

**生成后自检**：放大看线宽是否一致、是否只有纯黑纯白两色；不合格就重新生成，不要硬跑管线。

## 阶段 2：线稿矢量化

```bash
python "<技能目录>/scripts/pipeline.py" \
  --stage lineart --outdir "<项目>/output" --name <素材名>
```

管线（脚本内部固定顺序，不要调换）：

1. **二值化 + 裁剪** — 阈值 200（AI 线稿细线是抗锯齿浅灰，灰度中位约 167，用 128 会整条切掉）
2. **位图断缝修复：方向判据桥接** — 只有短碎片发起；桥连线须与参照侧笔画局部走向 |cos| ≥ 0.8 才连（沿笔画走向续接，不是"距离近就连"）
3. 骨架化 + 剪毛刺 + 去小连通域
4. **统一宽度重渲染** — 全体线条统一 2.0px，骨架拓扑零损失
5. 再骨架化 → 追踪点列（结点处自动拆笔）
6. **合并点列**（≤2px，短笔画发起 + 单趟贪心）
7. **降低线曲率**（点列级：顶点到弦垂距 <0.75px 拉向弦中点，3 轮）——必须在合并之后
8. **删除孤立噪点**（5px 内无其它笔画的点）——必须放最后
9. **拟合** — RDP 抽稀 → 拐角检测 → DP 最优分段 → 基元分级 L/A/C → 单一 path 输出

## 阶段 3：颜色填充

```bash
python "<技能目录>/scripts/pipeline.py" --stage fill --outdir "<项目>/output" --name <素材名>
```

前置：`<name>_normmask.png` + `<name>_lineart.svg`（阶段 2 产物）+ 原彩图 `original.jpg`。

- **配准**：线稿与彩图尺寸/构图不同，先做 FFT 互相关配准（墨迹 vs 梯度），参数缓存在 `output/_regist.json`
- **取色**：区域内部（到边界 >1.5px）像素的**中位数**（鲁棒，抗线稿污染），不限调色板
- **渐变**：大区按亮度分挡分层过渡 + 整区打底
- **上色路线固定为 B+C**：B（骨架环面遍历）作主基底画闭合区，C（泛洪分区 + 边缘吸附）作底层
  补齐 B 未覆盖的区域（背景/开环），线稿在最上层。这是定稿结构，正式版不提供路线切换

输出层序：白底 → `<g id="fills-base">` 补齐 → `<g id="fills-main">` 主基底 → `<path id="lineart">` 线稿。

## 一条命令跑全部

```bash
python "<技能目录>/scripts/pipeline.py" --stage all --outdir "<项目>/output" --name <素材名>
```

固定流程，无需（也无法）指定算法参数。全流程约 160s（线稿 ~80s + 上色 ~80s）。

## 命令行（正式版只暴露作业控制参数）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage` | all | `lineart`（只出线稿）/ `fill`（只上色，复用已有线稿产物）/ `all` |
| `--name` | out | 产物名前缀，用于区分不同任务 |
| `--outdir` | `./output` | 输出目录 |
| `--src-lineart` / `--src-color` | 自动探测 | 输入文件 |

**算法参数（二值化阈值、桥接强度、合并/降曲率阈值、渐变挡位、上色路线等）全部固定为
定稿配置，正式版不提供调参入口。** 流程与阈值都是实测定稿值：阈值 200、统一线宽 2px、
桥接 gap=3/长侧/cos≥0.8、合并 ≤2px、降曲率垂距 0.75px、孤立噪点 5px、渐变 3 挡、
上色固定 B+C。

## 常见坑

- **阶段 1 用了本地 CV 抽线稿**：必被阴影/噪点污染，完整度差一个档次。必须用生图工具重绘。
- **生图提示词没声明线条归一化**：粗细不均 → 骨架化分叉/毛刺 → 基元数暴涨。
- **二值化阈值用 128**：细线是抗锯齿浅灰像素，会整条切掉。用 180–200。
- **输入文件没按约定命名**：线稿 `lineart_src.png`、彩图 `original.jpg`，或显式传 `--src-lineart/--src-color`。
- **线稿是重绘版、彩图是原图**：不配准会整片错位（脚本自动做 FFT 互相关配准并缓存，别手改 `_regist.json`）。

## 环境要求

- Python 3.8+，需装 `numpy` `scipy` `scikit-image` `Pillow` `matplotlib`（`pip install -r requirements.txt`）
- 不依赖 potrace / vtracer：矢量化引擎为自带的自研实现（`scripts/fit_curve.py` + `scripts/lineart.py` + `scripts/trace.py`）
- 阶段 1 的生图依赖 ImageGen 工具（图生图 + `input_fidelity=high`）

## 交付

- 产物按 `--name` 命名：`<name>_lineart.svg`、`<name>_normmask.png`、`<name>_colored.svg`（**正式成品**）、`<name>_check.png`（对照图）
- 交付前把产物登记进 `output/README.md`
- 回复开头明确列出本轮交付物的**文件路径**

## 附：脚本资源

- `scripts/pipeline.py` — 正式流程主程序（`--stage lineart|fill|all`）
- `scripts/lineart.py` — 骨架化 / 追踪 / 拟合封装
- `scripts/fit_curve.py` — 最少基元曲线还原库（L/A/C 分级 + DP 最优分段 + G1 修正）
- `scripts/trace.py` — 闭合轮廓追踪与开放笔画 RDP（注意与标准库 `trace` 同名，脚本已强制优先导入同目录版本）
- `references/lineart_prompt.md` — 阶段 1 生图提示词模板（含线条归一化条款）
