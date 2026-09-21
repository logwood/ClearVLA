"""Insert the stage-reflection slide into the user's current review deck.

The source deck is intentionally treated as the source of truth here.  This
script adds one native, editable slide and moves it between the existing
LIBERO diagnosis page (page 5 in the current order) and the expansion-order
page.  It does not regenerate or reorder the rest of the deck, which keeps
manual edits made in PowerPoint intact.

The output path is separate from the source by default so the source can stay
open while the slide is being reviewed.  Run it only after the source deck is
closed if the output is going to replace the user's working file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(r"C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts")
DEFAULT_SOURCE = ROOT / "artifacts" / "clearvla_progress_review_20260913_coherent_mainline.pptx"
DEFAULT_OUTPUT = ROOT / "artifacts" / "clearvla_progress_review_20260913_coherent_mainline_with_reflection.pptx"

# Evidence files are deliberately stills here.  The adjacent CALVIN pages
# already carry the playable clips; this page uses end-of-clip frames to
# explain the contrast without replaying a video a second time.  The fallback
# paths keep the script usable if the derived plates are cleaned up later.
IMG_SINGLE = ROOT / "artifacts" / ".ppt_assets" / "reflection_single_end.png"
IMG_MULTI = ROOT / "artifacts" / ".ppt_assets" / "reflection_multi_end.png"
IMG_SINGLE_FALLBACK = ROOT / "artifacts" / "server_media" / "calvin_server_drawer_e4_245.png"
IMG_MULTI_FALLBACK = ROOT / "artifacts" / "server_media" / "calvin_server_multitask_seq05_360.png"
VID_COLOR_FAILURE = ROOT / "new_logs" / "current" / "calvin" / "closed_loop_videos" / "push_blue_block_right_k4_altpos_failure_360steps.mp4"
IMG_COLOR_FAILURE = ROOT / "artifacts" / ".ppt_assets" / "calvin_push_failure_altpos.png"

NAVY = RGBColor(8, 17, 32)
INK = RGBColor(14, 27, 48)
PANEL = RGBColor(17, 33, 56)
PANEL_ALT = RGBColor(24, 46, 75)
WHITE = RGBColor(244, 248, 252)
MUTED = RGBColor(165, 188, 214)
DIM = RGBColor(111, 139, 169)
CYAN = RGBColor(76, 211, 225)
ORANGE = RGBColor(255, 177, 83)
GREEN = RGBColor(93, 216, 155)
RED = RGBColor(243, 111, 111)
GRID = RGBColor(51, 77, 108)
FONT_LATIN = "Calibri"
FONT_CN = "Microsoft YaHei"
W, H = 13.333, 7.5


def _set_fill(shape, color: RGBColor, transparency: int | None = None) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if transparency is not None:
        try:
            shape.fill.transparency = transparency
        except Exception:
            pass


def set_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = NAVY


def shape_rect(slide, x, y, w, h, fill, radius=0.0, line=None, line_width=0.8):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    _set_fill(shape, fill)
    if radius:
        # Rounded-rectangle corner radius is controlled by the preset.  Keep
        # the argument for call-site readability and style parity.
        pass
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_width)
    return shape


def shape_oval(slide, x, y, w, h, fill, line=None, line_width=0.8):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    _set_fill(shape, fill)
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_width)
    return shape


def add_line(slide, x1, y1, x2, y2, color=GRID, width=0.8):
    line = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color
    line.line.width = Pt(width)
    return line


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=12,
    color=WHITE,
    bold=False,
    font=FONT_CN,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    # Explicit East-Asian font keeps Chinese text stable when opened on a
    # machine whose theme defaults differ from the source deck.
    try:
        run._r.get_or_add_rPr().set("{http://schemas.openxmlformats.org/drawingml/2006/main}lang", "zh-CN")
    except Exception:
        pass
    return box


def add_header(slide, kicker, title, subtitle, page_label="05", accent=ORANGE):
    shape_rect(slide, 0, 0, W, 0.07, accent)
    add_text(slide, kicker.upper(), 0.58, 0.27, 2.80, 0.18, 9.0, accent, True, FONT_LATIN)
    add_text(slide, title, 0.58, 0.55, 11.1, 0.46, 25.0, WHITE, True, FONT_CN)
    add_text(slide, subtitle, 0.60, 1.16, 11.4, 0.28, 11.5, MUTED, False, FONT_CN)
    add_text(slide, page_label, 12.24, 0.27, 0.52, 0.18, 9.0, DIM, True, FONT_LATIN, PP_ALIGN.RIGHT)


def add_footer(slide, source, note):
    add_line(slide, 0.55, 7.08, 12.78, 7.08, GRID, 0.6)
    add_text(slide, f"source · {source}", 0.58, 7.13, 9.7, 0.16, 7.7, DIM, False, FONT_LATIN)
    add_text(slide, note, 9.15, 7.13, 3.58, 0.16, 7.7, DIM, False, FONT_CN, PP_ALIGN.RIGHT)


def pill(slide, text, x, y, w, fill, color=NAVY, size=8.8):
    shape_rect(slide, x, y, w, 0.28, fill, 0.08)
    add_text(slide, text, x + 0.06, y + 0.045, w - 0.12, 0.16, size, color, True, FONT_CN, PP_ALIGN.CENTER)


def add_image_cover(slide, path: Path, x, y, w, h):
    if not path.exists():
        shape_rect(slide, x, y, w, h, PANEL_ALT, 0.06, GRID, 0.7)
        add_text(slide, "image unavailable", x, y + h / 2 - 0.08, w, 0.16, 8.5, MUTED, False, FONT_LATIN, PP_ALIGN.CENTER)
        return None
    with Image.open(path) as im:
        iw, ih = im.size
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    target = w / h
    source = iw / ih
    if source > target:
        pic.crop_left = pic.crop_right = (1.0 - target / source) / 2.0
    elif source < target:
        pic.crop_top = pic.crop_bottom = (1.0 - source / target) / 2.0
    frame = shape_rect(slide, x, y, w, h, NAVY, 0.05, GRID, 0.6)
    try:
        slide.shapes._spTree.remove(frame._element)
        slide.shapes._spTree.insert(2, frame._element)
    except Exception:
        pass
    return pic


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def add_video_card(slide, poster: Path, video: Path, label: str, detail: str,
                   x, y, w, h, accent=CYAN):
    """Embed a native movie with a caption below the playable area."""
    if not video.exists():
        raise FileNotFoundError(video)
    movie_h = max(0.86, h - 0.58)
    slide.shapes.add_movie(
        str(video),
        Inches(x), Inches(y), Inches(w), Inches(movie_h),
        poster_frame_image=str(poster) if poster.exists() else None,
        mime_type="video/mp4",
    )
    shape_rect(slide, x, y + movie_h, w, 0.58, INK, 0.0, INK, 0.1)
    shape_rect(slide, x, y + movie_h, 0.06, 0.58, accent)
    add_text(slide, label, x + 0.14, y + movie_h + 0.10, w - 0.28, 0.18, 10.8, WHITE, True, FONT_CN)
    add_text(slide, f"内嵌视频 · {detail}", x + 0.14, y + movie_h + 0.32, w - 0.28, 0.15, 8.3, MUTED, False, FONT_CN)


def draw_arrow(slide, x1, y1, x2, y2, color=GRID, width=1.0):
    add_line(slide, x1, y1, x2, y2, color, width)
    tri = slide.shapes.add_shape(
        MSO_SHAPE.ISOSCELES_TRIANGLE,
        Inches(x2 - 0.055), Inches(y2 - 0.065), Inches(0.11), Inches(0.13),
    )
    _set_fill(tri, color)
    tri.line.color.rgb = color
    tri.rotation = 90


def build_reflection_slide(prs: Presentation):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    set_bg(slide)
    add_header(
        slide,
        "阶段反思",
        "颜色词替换后，变化停在哪一层？",
        "固定视觉与历史，只替换指令；对比方向词、颜色词和对象读取。",
        page_label="05",
        accent=CYAN,
    )

    # One evidence field: a distinct full rollout for the color instruction on
    # the left and a compact fixed-observation readout on the right.  The page
    # records what the probe actually shows before naming the hypothesis.
    shape_rect(slide, 0.72, 1.70, 5.12, 4.78, PANEL, 0.10, GRID, 0.7)
    add_video_card(
        slide,
        IMG_COLOR_FAILURE,
        VID_COLOR_FAILURE,
        "颜色指令 · 未完成",
        "go push the blue block right · 360 步 · 30.6 s",
        0.98,
        2.00,
        4.60,
        2.56,
        RED,
    )
    add_text(slide, "视频：颜色指令未完成", 1.04, 4.82, 2.58, 0.20, 11.0, CYAN, True, FONT_CN)
    add_line(slide, 1.04, 5.12, 5.48, 5.12, GRID, 0.7)
    probe_rows = [
        ("左 ↔ 右", "S 变化 0.285–0.294", CYAN),
        ("红 / 粉 / 蓝", "S 变化 0.0058–0.0140", ORANGE),
        ("对象表征 / 读取", "语言替换最大差异 0.0", RED),
    ]
    for i, (label, value, col) in enumerate(probe_rows):
        yy = 5.28 + i * 0.36
        shape_oval(slide, 1.06, yy + 0.03, 0.13, 0.13, col)
        add_text(slide, label, 1.30, yy, 1.62, 0.18, 9.9, WHITE, True, FONT_CN)
        add_text(slide, value, 3.00, yy, 2.30, 0.18, 9.2, col, True, FONT_CN, PP_ALIGN.RIGHT)

    shape_rect(slide, 6.10, 1.70, 6.52, 4.78, PANEL_ALT, 0.10, GRID, 0.7)
    add_text(slide, "探针结果", 6.42, 2.02, 2.20, 0.22, 13.5, CYAN, True, FONT_CN)
    add_text(slide, "同一画面，只替换一句指令", 6.42, 2.40, 5.72, 0.24, 12.0, WHITE, True, FONT_CN)
    add_text(slide, "L = 语言条件　·　S = 公共意图载体", 6.42, 2.72, 5.72, 0.18, 9.2, MUTED, False, FONT_CN)

    # A single path keeps the suspected responsibility visible without
    # turning the page into another full-network diagram.
    node_specs = [
        ("语言条件 L", "T5 / 指令", 6.42, 3.06, 1.58, 0.84, CYAN),
        ("S", "公共意图载体", 8.56, 3.06, 1.46, 0.84, ORANGE),
        ("对象指针 K", "颜色 → 对象槽位", 10.62, 3.06, 1.66, 0.84, GREEN),
    ]
    for title, body, x, y, w, h, col in node_specs:
        shape_rect(slide, x, y, w, h, PANEL, 0.07, GRID, 0.7)
        shape_rect(slide, x, y, 0.07, h, col)
        add_text(slide, title, x + 0.18, y + 0.14, w - 0.26, 0.20, 11.0, WHITE, True, FONT_CN)
        add_text(slide, body, x + 0.18, y + 0.46, w - 0.26, 0.17, 8.8, MUTED, False, FONT_CN)
    draw_arrow(slide, 8.08, 3.48, 8.46, 3.48, GRID, 0.9)
    draw_arrow(slide, 10.06, 3.48, 10.52, 3.48, GRID, 0.9)

    add_text(slide, "方向词：S 有明显变化", 6.42, 4.28, 2.45, 0.20, 10.5, CYAN, True, FONT_CN)
    add_text(slide, "颜色词：S 变化很小", 8.92, 4.28, 2.35, 0.20, 10.5, ORANGE, True, FONT_CN)
    add_text(slide, "对象表征 / 读取：语言替换最大差异 0.0", 6.42, 4.76, 5.74, 0.20, 9.8, RED, True, FONT_CN)
    add_text(slide, "视觉对象路径仍有变化：对象表征 RMS 0.115–0.144", 6.42, 5.02, 5.74, 0.18, 9.5, GREEN, True, FONT_CN)

    # Treat the L → S → K sentence as the actual inspection path, not as a
    # conclusion.  The two checkpoints make the uncertainty explicit.
    shape_rect(slide, 6.42, 5.36, 5.76, 0.74, PANEL, 0.06, GRID, 0.6)
    add_text(slide, "检查路径", 6.62, 5.48, 0.84, 0.17, 9.2, MUTED, True, FONT_CN)
    add_text(slide, "L → S", 7.58, 5.47, 0.70, 0.18, 10.2, CYAN, True, FONT_CN)
    add_text(slide, "颜色词是否进入意图载体", 8.30, 5.47, 2.20, 0.18, 9.3, WHITE, False, FONT_CN)
    add_text(slide, "S → K", 7.58, 5.76, 0.70, 0.18, 10.2, ORANGE, True, FONT_CN)
    add_text(slide, "意图是否继续指向对象槽位", 8.30, 5.76, 2.68, 0.18, 9.3, WHITE, False, FONT_CN)
    add_footer(slide, "new_logs/current/calvin/object_binding_20260911.json · docs/research/CURRENT_MAINLINE_ISSUES.md", "反思 · L / S 绑定")
    return slide


def _slide_text(slide) -> str:
    chunks: list[str] = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text:
            chunks.append(shape.text)
    return "\n".join(chunks)


def insert_after_libero_diagnosis(prs: Presentation) -> None:
    # Locate the current LIBERO diagnosis by its title instead of relying on
    # a stale numeric page label.  This keeps the insertion beside the page
    # the user edited even if the visible section numbers remain historical.
    insertion_index = None
    for idx, slide in enumerate(prs.slides):
        text = _slide_text(slide)
        if "问题按出现顺序排查" in text or "起点问题研究" in text:
            insertion_index = idx + 1
            break
    if insertion_index is None:
        # The current deck has this page in position five (zero-based index 4).
        insertion_index = 5

    build_reflection_slide(prs)
    sld_ids = prs.slides._sldIdLst
    # The presentation roster contains p:sldId elements, not the slide's
    # p:sld element itself.  The newly-added roster entry is the last one.
    element = sld_ids[-1]
    sld_ids.remove(element)
    sld_ids.insert(insertion_index, element)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Output must be separate from the source while the deck may be open")
    output.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation(str(source))
    if len(prs.slides) < 5:
        raise RuntimeError(f"Expected at least five slides, found {len(prs.slides)}")
    insert_after_libero_diagnosis(prs)
    prs.save(str(output))
    print(output)


if __name__ == "__main__":
    main()
