# 阶段 1：位图 → 线稿 的生图提示词模板

本地 CV 抽线稿（黑帽变换 / XDoG / 照度平坦化）在插画类图上会被阴影、噪点、抗锯齿
污染，完整度差一个档次。**必须用生图工具（ImageGen 图生图）重绘为线稿。**

调用要点：
- 图生图模式，传入原图
- `input_fidelity=high`（保证结构/构图不漂移）
- 尺寸长边 1024（太小丢细节，太大只是拖慢后续骨架化）
- **提示词必须包含"线条归一化"条款**，否则后续矢量化会退化

---

## 必须声明的四条（缺一不可）

1. **线条归一化 / 统一粗细** — 全图所有线条使用**同一个恒定线宽**，无粗细变化、
   无笔锋、无压感、无毛笔飞白。这是硬要求，直接决定骨架化质量。
2. **纯黑白二值** — 只有纯黑线条 + 纯白背景。禁止灰阶、抗锯齿灰边、阴影、
   渐变、涂色、网点、纹理。
3. **轮廓闭合、结构完整** — 保留原图全部结构与细节，不新增、不删减元素；
   封闭区域的轮廓必须闭合。
4. **白底、干净** — 纯白背景，无噪点、无杂散短线。

---

## 模板 A：英文（推荐，生图模型对英文响应更稳）

```
Convert this illustration into clean monochrome line art for vectorization.

LINE NORMALIZATION (critical):
- Every line must have the SAME constant stroke width. No thick-thin variation,
  no tapered ends, no brush pressure, no calligraphic strokes.
- Uniform, even, mechanically consistent line weight across the entire image.

Requirements:
- Pure black lines (#000000) on a pure white background (#FFFFFF), strictly binary.
- No grayscale, no anti-aliased gray edges, no shading, no hatching, no texture,
  no gradient, no fill color, no shadows.
- Closed and complete contours; preserve ALL structures and details of the original.
- Do not add or remove any element. Keep the exact same composition and pose.
- Clean single-pixel-like strokes, no noise, no stray marks.
```

## 模板 B：中文

```
把这张插画转成干净的单色线稿，用于矢量化。

线条归一化（关键要求）：
- 全图所有线条使用完全相同的恒定线宽，不允许粗细变化、笔锋、压感或毛笔飞白。
- 线宽均匀、机械一致，贯穿整幅图。

其它要求：
- 纯黑线条（#000000）+ 纯白背景（#FFFFFF），严格二值，无灰阶、无抗锯齿灰边。
- 无阴影、无排线、无纹理、无渐变、无填色、无网点。
- 轮廓闭合且完整，保留原图全部结构与细节。
- 不新增、不删减任何元素，构图与姿态完全一致。
- 线条干净，无噪点、无杂散短线。
```

## 模板 C：极简版（模型较强时用）

```
Monochrome line art with perfectly uniform stroke width (line normalization:
every stroke the same constant width, no variation). Pure black lines on pure
white background, strictly binary, no shading or gradient or anti-aliasing.
Closed clean contours, preserve all details, same composition.
```

---

## 生成后的自检（不合格就重生成，别硬跑管线）

- 放大看线条是否**粗细一致**（有粗细变化 → 骨架化会分叉/毛刺，基元数暴涨）
- 是否**只有纯黑纯白**（有灰边 → 阈值 200 二值化时会断线）
- 原图的关键结构是否都在（生图会"脑补"，结构与原图不符时后续配准/取色都会错）

质检通过后另存为 `output/lineart_src.png`，再进入阶段 2。

> 补充：若生成结果线条偏细导致二值化后断线，把 `--thresh` 从 200 降到 180；
> 若噪点变多则升到 210–220。不要用 128。
