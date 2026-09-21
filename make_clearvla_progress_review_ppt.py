from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(r"C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts")
OUT = ROOT / "artifacts" / "clearvla_progress_review_20260913_slide1_what_we_did.pptx"

# Evidence assets already present in the workspace.
# Full-length rollouts are used for the presentation surface.  Short causal
# and timing clips remain in the workspace as source evidence, but are not
# promoted to playable cards.
IMG_LIBERO_FULL = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e8_180/agent_contact_uniform12.png"
IMG_LIBERO_FULL_E4 = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e4_180/contact_sheet_uniform12.png"
VID_LIBERO_FULL = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e8_180/task_0000_episode_000.mp4"
VID_LIBERO_FULL_E4 = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e4_180/task_0000_episode_000.mp4"
IMG_HANDOFF_EXPERT = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2_expert_contact_sheet.jpg"
IMG_HANDOFF_COLD = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2_cold_contact_sheet.jpg"
VID_HANDOFF_EXPERT = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2_videos/expert_prefix/task_0000_episode_000.mp4"
VID_HANDOFF_COLD = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2_videos/cold_start/task_0000_episode_000.mp4"
IMG_STACK_0 = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/step_0000.png"
IMG_STACK_100 = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/step_0100.png"
IMG_STACK_200 = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/step_0200.png"
IMG_STACK_400 = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/step_0400.png"
VID_STACK = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/rollout.mp4"
VID_STACK_APPROACH = ROOT / "runs/stackcube_v2_eval_20260910/approach_e8_seed1000001/rollout.mp4"
# A few dense-but-readable evidence strips used to keep the later pages
# grounded in actual rollouts rather than abstract module boxes.
IMG_ROLLOUT_GRID = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e8_180/contact_sheet.png"
IMG_GRIPPER_KEY = ROOT / "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e8_180/gripper_key.png"
# Retarget / release-centered evidence is presented as a compact quantitative
# trace.  The source workspace contains 8/32-frame probe contact sheets; they
# stay in the audit folders and are intentionally not promoted into the deck.
RETARGET_ASSET_DIR = ROOT / "artifacts" / ".ppt_assets"
IMG_RETARGET_ALL = RETARGET_ASSET_DIR / "retarget_summary.png"
IMG_RETARGET_BASE = RETARGET_ASSET_DIR / "retarget_baseline.png"
IMG_RETARGET_PLUS = RETARGET_ASSET_DIR / "retarget_plus3cm.png"
IMG_RETARGET_MINUS = RETARGET_ASSET_DIR / "retarget_minus3cm.png"

# CALVIN clips are shown as a separate cross-environment evidence page.  The
# posters are real rollout contact sheets, so the page stays visual even in
# viewers that do not autoplay native movies.
VID_CALVIN_PUSH = ROOT / "new_logs/current/calvin/closed_loop_videos/push_blue_block_right_k4_success_73steps.mp4"
IMG_CALVIN_PUSH = ROOT / "new_logs/current/calvin/abc_d_expanded_exp1_20260905/pilot_rollout_best_k4_s180/contact_sheets/sequence_01__go_push_the_blue_block_right.png"
VID_CALVIN_PUSH_FAILURE = ROOT / "new_logs/current/calvin/closed_loop_videos/push_blue_block_right_k4_altpos_failure_360steps.mp4"
IMG_CALVIN_PUSH_FAILURE = ROOT / "artifacts/.ppt_assets/calvin_push_failure_altpos.png"
VID_CALVIN_DRAWER_K4 = ROOT / "new_logs/current/calvin/open_drawer_v2_20260905/closed_loop_e4_best_k4/open_drawer_k4.mp4"
IMG_CALVIN_DRAWER_K4 = ROOT / "new_logs/current/calvin/open_drawer_v2_20260905/closed_loop_e4_best_k4/contact_sheet.png"
VID_CALVIN_SLIDER = ROOT / "new_logs/current/calvin/abc_d_expanded_exp1_20260905/pilot_rollout_best_k4_s180/videos_typical_prompt/sequence_05/push the sliding door to the left side.mp4"
IMG_CALVIN_SLIDER = ROOT / "new_logs/current/calvin/abc_d_expanded_exp1_20260905/pilot_rollout_best_k4_s180/contact_sheets/sequence_05__push_the_sliding_door_to_the_left_side.png"
VID_CALVIN_DRAWER_SEQ09 = ROOT / "new_logs/current/calvin/abc_d_expanded_exp1_20260905/pilot_rollout_best_k4_s180/videos_typical_prompt/sequence_09/pull the handle to open the drawer.mp4"
IMG_CALVIN_DRAWER_SEQ09 = ROOT / "new_logs/current/calvin/abc_d_expanded_exp1_20260905/pilot_rollout_best_k4_s180/contact_sheets/sequence_09__pull_the_handle_to_open_the_drawer.png"
# Additional CALVIN success clips kept distinct from the opening evidence
# slide.  They make the CALVIN page a real set of examples rather than four
# copies of one rollout.
VID_CALVIN_DRAWER_E4 = ROOT / "new_logs/current/calvin/open_drawer_v2_20260905/closed_loop_e4_best/open_drawer_e4_best_success.mp4"
IMG_CALVIN_DRAWER_E4 = ROOT / "new_logs/current/calvin/open_drawer_v2_20260905/closed_loop_e4_best/static_contact_sheet_2fps.png"
VID_CALVIN_SCHEMA30 = ROOT / "new_logs/current/calvin/historical_rollouts/open_drawer_schema30_fdfix_latest_retry_20260903/open_drawer_replay.mp4"
IMG_CALVIN_SCHEMA30 = ROOT / "new_logs/current/calvin/historical_rollouts/open_drawer_schema30_fdfix_latest_retry_20260903/contact_sheet_2fps.png"

# CALVIN replays copied from senwang-server for this review.  The server set
# gives us both completed single-task clips and full 360-step multi-task
# scenes, so the deck can show the hand-off into multi-task without recycling
# one short local clip.
SERVER_MEDIA_DIR = ROOT / "artifacts/server_media"
VID_SERVER_SLIDER_SEQ05 = SERVER_MEDIA_DIR / "calvin_server_slider_seq05.mp4"
IMG_SERVER_SLIDER_SEQ05 = SERVER_MEDIA_DIR / "calvin_server_slider_seq05.png"
VID_SERVER_DRAWER_E4_245 = SERVER_MEDIA_DIR / "calvin_server_drawer_e4_245.mp4"
IMG_SERVER_DRAWER_E4_245 = SERVER_MEDIA_DIR / "calvin_server_drawer_e4_245.png"
VID_SERVER_DRAWER_TARGETED_163 = SERVER_MEDIA_DIR / "calvin_server_drawer_targeted_163.mp4"
IMG_SERVER_DRAWER_TARGETED_163 = SERVER_MEDIA_DIR / "calvin_server_drawer_targeted_163.png"
VID_SERVER_DRAWER_SEQ09 = SERVER_MEDIA_DIR / "calvin_server_drawer_seq09.mp4"
IMG_SERVER_DRAWER_SEQ09 = SERVER_MEDIA_DIR / "calvin_server_drawer_seq09.png"
IMG_CALVIN_PUSH_POSTER = SERVER_MEDIA_DIR / "calvin_push_success_73.png"
VID_SERVER_DRAWER_K4_69 = SERVER_MEDIA_DIR / "calvin_server_drawer_k4_69.mp4"
IMG_SERVER_DRAWER_K4_69 = SERVER_MEDIA_DIR / "calvin_server_drawer_k4_69.png"
VID_SERVER_MULTI_SEQ05_360 = SERVER_MEDIA_DIR / "calvin_server_multitask_seq05_360.mp4"
IMG_SERVER_MULTI_SEQ05_360 = SERVER_MEDIA_DIR / "calvin_server_multitask_seq05_360.png"
VID_SERVER_MULTI_SEQ07_360 = SERVER_MEDIA_DIR / "calvin_server_multitask_seq07_360.mp4"
IMG_SERVER_MULTI_SEQ07_360 = SERVER_MEDIA_DIR / "calvin_server_multitask_seq07_360.png"

CAUSAL_JSON = ROOT / "artifacts/libero_causal_ab_20260910/v3_audit.json"
TIMING_JSON = ROOT / "artifacts/libero_timing_20260911/timing_probe_v1.json"
HANDOFF_JSON = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2.json"
RELEASE_AUDIT = ROOT / "artifacts/libero_release_first_20260913/r4_releasefirst/audit.json"
STACK_SUMMARY = ROOT / "runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/summary.txt"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def parse_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in raw:
            k, v = raw.split("=", 1)
            out[k.strip()] = v.strip()
    return out


causal = load_json(CAUSAL_JSON)
causal_a = causal["reports"][0]
causal_b = causal["reports"][1]
timing = load_json(TIMING_JSON)
timing_case = next(iter(timing["cases"].values()))
handoff = load_json(HANDOFF_JSON)
handoff_case = next(iter(handoff["cases"].values()))
release = load_json(RELEASE_AUDIT)
release_run = release["runs"][0]
release_val = release_run["latest_epoch_metrics"]["val"]
stack = parse_kv(STACK_SUMMARY)


def num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# Palette: cool technical field with sparse warm evidence accents.
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
BLACK = RGBColor(0, 0, 0)
# Calibri is present in the default Office theme and travels more reliably
# than the newer Aptos face across older PowerPoint installations.
FONT_LATIN = "Calibri"
FONT_CN = "Microsoft YaHei"
W, H = 13.333, 7.5


def _asset_font(size: int, bold: bool = False):
    """Load a stable local font for the small quantitative evidence plates."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _ensure_retarget_assets():
    """Create quantitative retarget plates instead of exposing short probes.

    The underlying audit has paired short sanity checks, but their frame grids
    are not useful presentation evidence.  These plates keep the measured
    release-row values and the matched-init-state contract visible without
    pretending that a handful of frames is a long rollout.
    """
    RETARGET_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    bg = (17, 33, 56)
    panel = (24, 46, 75)
    grid = (51, 77, 108)
    white = (244, 248, 252)
    muted = (165, 188, 214)
    dim = (111, 139, 169)
    cyan = (76, 211, 225)
    orange = (255, 177, 83)
    green = (93, 216, 155)

    def rounded(draw, box, fill, outline=grid, radius=18, width=2):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    title_font = _asset_font(28, True)
    label_font = _asset_font(22, True)
    value_font = _asset_font(24, True)
    small_font = _asset_font(18, False)
    tiny_font = _asset_font(16, False)

    # Wide summary plate used where a visual anchor is needed.
    im = Image.new("RGB", (1600, 520), bg)
    d = ImageDraw.Draw(im)
    rounded(d, (8, 8, 1592, 512), panel, outline=grid, radius=22, width=2)
    d.text((38, 30), "PAIRED REPLAY / ACTION BOUNDARY", fill=cyan, font=small_font)
    d.text((38, 72), "release-row readout", fill=white, font=title_font)
    d.text((38, 116), "same initial state · target shift ±30 mm", fill=muted, font=small_font)
    left, right, zero = 370, 1450, 1010
    scale = 420
    for tick, text in [(-1.0, "−1.0"), (-0.5, "−0.5"), (0.0, "0"), (0.3, "+0.3")]:
        xx = int(zero + tick * scale)
        d.line((xx, 160, xx, 426), fill=grid, width=2)
        d.text((xx - 22, 438), text, fill=dim, font=tiny_font)
    d.line((left, 160, right, 160), fill=grid, width=2)
    rows = [("baseline", 0.2265, cyan), ("+3 cm", -0.4698, orange), ("−3 cm", -1.0, green)]
    for i, (lab, val, col) in enumerate(rows):
        yy = 218 + i * 82
        d.text((54, yy - 16), lab, fill=white, font=label_font)
        d.line((left, yy + 8, right, yy + 8), fill=(37, 60, 88), width=12)
        end = int(zero + val * scale)
        d.line((zero, yy + 8, end, yy + 8), fill=col, width=12)
        d.ellipse((end - 11, yy - 3, end + 11, yy + 19), fill=col)
        sign = "+" if val > 0 else ""
        d.text((1480, yy - 15), f"{sign}{val:.4f}", fill=col, font=value_font)
    d.text((54, 466), "release row 85", fill=muted, font=tiny_font)
    d.text((1020, 466), "4/4 paired cases stationary", fill=green, font=tiny_font)
    im.save(IMG_RETARGET_ALL, format="PNG", optimize=True)

    # Three compact plates keep the individual cards concrete and distinct.
    for path, lab, val, col in [
        (IMG_RETARGET_BASE, "baseline", 0.2265, cyan),
        (IMG_RETARGET_PLUS, "+3 cm", -0.4698, orange),
        (IMG_RETARGET_MINUS, "−3 cm", -1.0, green),
    ]:
        card = Image.new("RGB", (920, 430), bg)
        cd = ImageDraw.Draw(card)
        rounded(cd, (8, 8, 912, 422), panel, outline=grid, radius=20, width=2)
        cd.text((34, 28), "RELEASE ROW 85", fill=dim, font=tiny_font)
        cd.text((34, 64), lab, fill=white, font=title_font)
        axis_left, axis_right, axis_zero = 150, 830, 510
        axis_scale = 320
        cd.line((axis_left, 224, axis_right, 224), fill=(37, 60, 88), width=14)
        cd.line((axis_zero, 172, axis_zero, 276), fill=grid, width=2)
        end = int(axis_zero + val * axis_scale)
        cd.line((axis_zero, 224, end, 224), fill=col, width=14)
        cd.ellipse((end - 14, 210, end + 14, 238), fill=col)
        sign = "+" if val > 0 else ""
        cd.text((34, 304), f"{sign}{val:.4f}", fill=col, font=value_font)
        cd.text((200, 312), "matched init state", fill=muted, font=small_font)
        cd.text((200, 350), "paired replay · same init state", fill=dim, font=tiny_font)
        card.save(path, format="PNG", optimize=True)


_ensure_retarget_assets()

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]
prs.core_properties.title = "ClearVLA · recent work progress review"
prs.core_properties.subject = "LIBERO / CALVIN / StackCube evidence review and next experiments"
prs.core_properties.author = "ClearVLA"
_EMBEDDED_VIDEOS: set[str] = set()
_POSTER_DIR = ROOT / "artifacts" / ".ppt_video_posters"
_POSTER_CACHE: dict[tuple[str, int, int], Path] = {}


def set_bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def shape_rect(slide, x, y, w, h, fill, radius=0.0, line_color=None, line_width=0.8):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line_color or fill
    s.line.width = Pt(line_width)
    if radius:
        try:
            s.adjustments[0] = min(0.18, radius)
        except Exception:
            pass
    return s


def shape_oval(slide, x, y, w, h, fill, line_color=None, line_width=0.8):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line_color or fill
    s.line.width = Pt(line_width)
    return s


def add_line(slide, x1, y1, x2, y2, color=GRID, width=1.0, dash=None):
    s = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    s.line.color.rgb = color
    s.line.width = Pt(width)
    if dash:
        try:
            s.line.dash_style = dash
        except Exception:
            pass
    return s


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=16,
    color=WHITE,
    bold=False,
    font=FONT_CN,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.04,
    italic=False,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_rich_text(slide, runs: Iterable[tuple[str, int, RGBColor, bool]], x, y, w, h, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = align
    for txt, size, color, bold in runs:
        r = p.add_run()
        r.text = txt
        r.font.name = FONT_CN
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
    return tb


def add_header(slide, kicker, title, page, subtitle=None, accent=CYAN):
    set_bg(slide)
    shape_rect(slide, 0, 0, W, 0.07, accent)
    add_text(slide, kicker.upper(), 0.55, 0.25, 4.9, 0.25, 9.5, accent, True, FONT_LATIN)
    add_text(slide, title, 0.55, 0.58, 11.85, 0.58, 25, WHITE, True)
    if subtitle:
        add_text(slide, subtitle, 0.58, 1.15, 11.7, 0.28, 11.5, MUTED)
    # Number from the actual slide order.  This keeps page labels correct when
    # evidence pages are inserted into an existing review without hand-editing
    # every later literal page number.
    actual_page = len(prs.slides)
    add_text(slide, f"{actual_page:02d}", 12.28, 0.26, 0.5, 0.23, 9.5, DIM, True, FONT_LATIN, PP_ALIGN.RIGHT)


def add_footer(slide, source=None, note=None):
    add_line(slide, 0.55, 7.08, 12.78, 7.08, GRID, 0.6)
    if source:
        add_text(slide, f"source · {source}", 0.58, 7.13, 9.5, 0.17, 7.7, DIM, False, FONT_LATIN)
    if note:
        add_text(slide, note, 9.1, 7.13, 3.65, 0.17, 7.7, DIM, False, FONT_CN, PP_ALIGN.RIGHT)


def pill(slide, text, x, y, w, fill, color=NAVY, size=9.5):
    shape_rect(slide, x, y, w, 0.28, fill, 0.08)
    add_text(slide, text, x + 0.06, y + 0.04, w - 0.12, 0.17, size, color, True, FONT_CN, PP_ALIGN.CENTER)


def bullet(slide, text, x, y, w, color=WHITE, size=14, accent=CYAN, h=0.38):
    shape_oval(slide, x, y + 0.11, 0.085, 0.085, accent)
    add_text(slide, text, x + 0.18, y, w - 0.18, h, size, color)


def metric_card(slide, label, value, x, y, w, accent=CYAN, sub="", h=0.92):
    shape_rect(slide, x, y, w, h, PANEL, 0.08, GRID, 0.7)
    add_text(slide, label.upper(), x + 0.15, y + 0.11, w - 0.3, 0.17, 8.5, DIM, True, FONT_LATIN)
    add_text(slide, value, x + 0.15, y + 0.32, w - 0.3, 0.31, 22, accent, True, FONT_LATIN)
    if sub:
        add_text(slide, sub, x + 0.15, y + h - 0.2, w - 0.3, 0.14, 8.2, MUTED, False, FONT_CN)


def add_image_cover(slide, path: Path, x, y, w, h, frame=True):
    if not path.exists():
        shape_rect(slide, x, y, w, h, PANEL, 0.08, GRID, 0.7)
        add_text(slide, "image unavailable", x, y + h / 2 - 0.1, w, 0.2, 9, MUTED, False, FONT_LATIN, PP_ALIGN.CENTER)
        return None
    try:
        with Image.open(path) as im:
            iw, ih = im.size
        target = w / h
        source = iw / ih
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
        if source > target:
            pic.crop_left = pic.crop_right = (1.0 - target / source) / 2.0
        elif source < target:
            pic.crop_top = pic.crop_bottom = (1.0 - source / target) / 2.0
        if frame:
            frame_shape = shape_rect(slide, x, y, w, h, RGBColor(0, 0, 0), 0.07, GRID, 0.7)
            # Move frame behind the picture when possible.
            try:
                slide.shapes._spTree.remove(frame_shape._element)
                slide.shapes._spTree.insert(2, frame_shape._element)
            except Exception:
                pass
        return pic
    except Exception:
        shape_rect(slide, x, y, w, h, PANEL, 0.08, GRID, 0.7)
        add_text(slide, "image unavailable", x, y + h / 2 - 0.1, w, 0.2, 9, MUTED, False, FONT_LATIN, PP_ALIGN.CENTER)
        return None


def add_image_contain(slide, path: Path, x, y, w, h, frame=True):
    """Place an evidence plate without cropping its axes or labels."""
    if not path.exists():
        return add_image_cover(slide, path, x, y, w, h, frame=frame)
    try:
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        xx, yy = x + (w - dw) / 2.0, y + (h - dh) / 2.0
        if frame:
            shape_rect(slide, x, y, w, h, PANEL, 0.07, GRID, 0.7)
        return slide.shapes.add_picture(str(path), Inches(xx), Inches(yy), width=Inches(dw), height=Inches(dh))
    except Exception:
        return add_image_cover(slide, path, x, y, w, h, frame=frame)


def file_uri(path: Path) -> str:
    return path.resolve().as_uri() if path.exists() else ""


def link_shape(shape, path: Path):
    uri = file_uri(path)
    if uri:
        try:
            shape.click_action.hyperlink.address = uri
        except Exception:
            pass
    return shape


def video_badge(slide, x, y, w, label, path: Path, duration="MP4"):
    # Static label only.  The movie object itself owns the play action; a
    # hyperlink overlay would force PowerPoint's Ctrl+click behaviour.
    badge = shape_rect(slide, x, y, w, 0.31, INK, 0.08, CYAN, 0.65)
    shape_oval(slide, x + 0.08, y + 0.07, 0.16, 0.16, CYAN)
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x + 0.125), Inches(y + 0.092), Inches(0.07), Inches(0.1))
    tri.fill.solid(); tri.fill.fore_color.rgb = NAVY; tri.line.color.rgb = NAVY
    tri.rotation = 90
    add_text(slide, f"{label} · {duration}", x + 0.31, y + 0.065, w - 0.38, 0.17, 8.7, WHITE, True, FONT_CN)
    return badge


def poster_for(image: Path, w: float, h: float) -> Path | None:
    """Make a center-cropped poster frame for a native movie shape."""
    if not image.exists():
        return None
    key = (str(image.resolve()), int(round(w * 100)), int(round(h * 100)))
    if key in _POSTER_CACHE:
        return _POSTER_CACHE[key]
    try:
        _POSTER_DIR.mkdir(parents=True, exist_ok=True)
        out = _POSTER_DIR / f"poster_{len(_POSTER_CACHE):02d}.png"
        with Image.open(image) as im:
            im = im.convert("RGB")
            target = w / h
            source = im.width / im.height
            if source > target:
                crop_w = int(round(im.height * target))
                left = max(0, (im.width - crop_w) // 2)
                im = im.crop((left, 0, left + crop_w, im.height))
            elif source < target:
                crop_h = int(round(im.width / target))
                top = max(0, (im.height - crop_h) // 2)
                im = im.crop((0, top, im.width, top + crop_h))
            im.save(out, format="PNG", optimize=True)
        _POSTER_CACHE[key] = out
        return out
    except Exception:
        return None


def video_card(slide, image: Path, video: Path, label: str, sub: str, x, y, w, h, accent=CYAN, duration=""):
    # Embed one native video object per source clip.  The movie object carries
    # PowerPoint's built-in ppaction://media click action, so the viewer can
    # click the video directly; no Ctrl+click hyperlink overlay is needed.
    video_key = str(video.resolve()) if video.exists() else ""
    if not video_key:
        raise FileNotFoundError(f"Missing video source: {video}")
    if video_key in _EMBEDDED_VIDEOS:
        raise RuntimeError(f"Video source reused on presentation surface: {video}")
    movie_h = max(0.72, h - 0.58)
    try:
        slide.shapes.add_movie(
            str(video),
            Inches(x), Inches(y), Inches(w), Inches(movie_h),
            poster_frame_image=str(poster_for(image, w, movie_h) or image) if image.exists() else None,
            mime_type="video/mp4",
        )
        _EMBEDDED_VIDEOS.add(video_key)
    except Exception as exc:
        raise RuntimeError(f"Could not embed video {video}: {exc}") from exc
    # Keep captions outside the movie rectangle so every point on the poster
    # remains a direct click target for the embedded media object.
    shape_rect(slide, x, y + movie_h, w, 0.58, INK, 0.0, INK, 0.1)
    add_text(slide, label, x + 0.14, y + movie_h + 0.11, w - 0.28, 0.18, 11, WHITE, True)
    detail = sub
    if duration:
        detail = f"{sub} · {duration}" if sub else duration
    add_text(slide, f"内嵌视频 · {detail}" if detail else "内嵌视频", x + 0.14, y + movie_h + 0.33, w - 0.28, 0.15, 8.5, MUTED, False, FONT_CN)


def image_card(slide, image: Path, label: str, sub: str, x, y, w, h, accent=CYAN, tag="证据图"):
    """A non-playable evidence card for diagnostic stills/contact sheets.

    Short probes are useful in the audit, but a 1–4 second clip is not a
    meaningful presentation demo.  This card keeps the source image and the
    conclusion while making it visually clear that the page is a diagnosis,
    not another replay.
    """
    shape_rect(slide, x, y, w, h, PANEL, 0.09, GRID, 0.7)
    image_h = max(0.62, h - 0.72)
    if image.parent == RETARGET_ASSET_DIR:
        add_image_contain(slide, image, x + 0.12, y + 0.12, w - 0.24, image_h, frame=True)
    else:
        add_image_cover(slide, image, x + 0.12, y + 0.12, w - 0.24, image_h, frame=True)
    pill(slide, tag, x + 0.24, y + 0.25, min(1.05, max(0.78, w * 0.28)), accent, NAVY, 8.2)
    shape_rect(slide, x + 0.12, y + h - 0.60, w - 0.24, 0.48, INK, 0.0, INK, 0.1)
    add_text(slide, label, x + 0.24, y + h - 0.49, w - 0.48, 0.17, 10.8, WHITE, True)
    add_text(slide, sub, x + 0.24, y + h - 0.27, w - 0.48, 0.14, 8.4, MUTED, False, FONT_CN)


def section_marker(slide, text, x, y, color=CYAN):
    shape_rect(slide, x, y, 0.08, 0.28, color)
    add_text(slide, text, x + 0.17, y - 0.01, 3.2, 0.27, 13, color, True)


def draw_arrow(slide, x1, y1, x2, y2, color=CYAN, width=1.2):
    add_line(slide, x1, y1, x2, y2, color, width)
    # Small triangle at the destination; avoids python-pptx arrowhead incompatibility.
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x2 - 0.05), Inches(y2 - 0.06), Inches(0.10), Inches(0.12))
    tri.fill.solid(); tri.fill.fore_color.rgb = color; tri.line.color.rgb = color
    tri.rotation = 90


def node(slide, title, body, x, y, w, h, fill, accent, title_size=13, body_size=9.5):
    shape_rect(slide, x, y, w, h, fill, 0.08, GRID, 0.7)
    shape_rect(slide, x, y, 0.07, h, accent)
    add_text(slide, title, x + 0.18, y + 0.14, w - 0.3, 0.23, title_size, WHITE, True)
    add_text(slide, body, x + 0.18, y + 0.46, w - 0.3, h - 0.55, body_size, MUTED)


def mini_bar(slide, label, value, max_value, x, y, w, color, value_text=None, h=0.18):
    add_text(slide, label, x, y - 0.02, 1.35, 0.18, 9.2, MUTED, False, FONT_CN)
    shape_rect(slide, x + 1.4, y, w - 1.95, h, GRID, 0.05)
    fill_w = max(0.02, (w - 1.95) * min(1.0, max(0.0, value / max_value)))
    shape_rect(slide, x + 1.4, y, fill_w, h, color, 0.05)
    add_text(slide, value_text if value_text is not None else f"{value:.3f}", x + w - 0.5, y - 0.03, 0.5, 0.19, 9.2, color, True, FONT_LATIN, PP_ALIGN.RIGHT)


def sparkline(slide, values, x, y, w, h, color=CYAN, baseline=None):
    if not values:
        return
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        hi = lo + 1
    pts = []
    for i, v in enumerate(values):
        px = x + (w * i / max(1, len(values) - 1))
        py = y + h - (v - lo) / (hi - lo) * h
        pts.append((px, py))
    if baseline is not None:
        by = y + h - (baseline - lo) / (hi - lo) * h
        add_line(slide, x, by, x + w, by, GRID, 0.6)
    for a, b in zip(pts, pts[1:]):
        add_line(slide, a[0], a[1], b[0], b[1], color, 2.0)
    for px, py in pts:
        shape_oval(slide, px - 0.045, py - 0.045, 0.09, 0.09, color)


# --- Slide 1: what we did ------------------------------------------------
s = prs.slides.add_slide(BLANK)
set_bg(s)
# Keep the supplied cover's quiet navy field and thin cyan frame, but use the
# page to state the work already carried out rather than a future work plan.
shape_rect(s, 0, 0, W, 0.10, CYAN)
shape_rect(s, 0, H - 0.10, W, 0.10, CYAN)
shape_rect(s, 0, 0.10, 0.10, H - 0.20, CYAN)
shape_rect(s, W - 0.10, 0.10, 0.10, H - 0.20, CYAN)
add_text(s, "9.14 组会", 0.72, 0.48, 11.90, 0.62, 38, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_text(s, "这段时间，实际改了什么", 0.84, 1.32, 7.18, 0.36, 22, WHITE, True, FONT_CN)

# A single vertical evidence spine keeps the three areas connected.  Each row
# has a concrete object line and a second line describing the implemented
# interface or boundary; no future-tense claims are used.
shape_rect(s, 1.02, 2.15, 0.06, 4.18, GRID)
what_we_did = [
    (
        "把三套环境接入可回放链路",
        "LIBERO / CALVIN / StackCube 的观测—动作—语言接口",
        "LIBERO 180 步；CALVIN 服务器回放；StackCube 400 步闭环回放",
        CYAN,
    ),
    (
        "把视觉、对象和历史条件串进网络主干",
        "视觉（DINO）/ 对象表征 / 语言、状态、历史条件",
        "G/S/W/P 主干；24 × 18 动作表示；先提议，再精化",
        ORANGE,
    ),
    (
        "把 residual SAC 接到 StackCube runner",
        "Soft Actor-Critic（SAC）；底层动作 + Δa",
        "环境、数据、replay、runner 已接入；残差只修正底层动作",
        GREEN,
    ),
]
row_y = 2.18
for i, (title, detail, subdetail, col) in enumerate(what_we_did):
    yy = row_y + i * 1.42
    shape_oval(s, 0.88, yy + 0.03, 0.34, 0.34, col)
    add_text(s, f"0{i + 1}", 0.88, yy + 0.095, 0.34, 0.14, 8.5, NAVY, True, FONT_LATIN, PP_ALIGN.CENTER)
    add_text(s, title, 1.42, yy - 0.01, 6.58, 0.28, 15.0, col, True, FONT_CN)
    add_text(s, detail, 1.42, yy + 0.38, 6.58, 0.24, 11.5, WHITE, True, FONT_CN)
    add_text(s, subdetail, 1.42, yy + 0.70, 6.58, 0.22, 9.8, MUTED, False, FONT_CN)
    if i < len(what_we_did) - 1:
        add_line(s, 1.42, yy + 1.13, 8.04, yy + 1.13, GRID, 0.7)

# Put the evidence beside the three workstreams, but split it by outcome.
# This keeps a success replay from being read as if it were a failed rollout.
shape_rect(s, 8.46, 1.34, 4.08, 5.62, PANEL, 0.10, GRID, 0.7)
add_text(s, "回放证据 · 按结果分开", 8.78, 1.60, 3.40, 0.24, 13.0, CYAN, True, FONT_CN)

# Verified completed replay (CALVIN sequence 09 / drawer).
pill(s, "已完成", 8.78, 1.96, 0.86, GREEN, NAVY, 8.4)
add_image_cover(s, IMG_SERVER_DRAWER_E4_245, 8.78, 2.30, 3.44, 1.14, frame=True)
add_text(s, "CALVIN · 开抽屉", 8.84, 3.58, 2.06, 0.18, 10.0, WHITE, True, FONT_CN)
add_text(s, "245 步 · 判定成功", 10.22, 3.58, 1.82, 0.18, 9.0, GREEN, True, FONT_CN, PP_ALIGN.RIGHT)
add_line(s, 8.78, 3.92, 12.22, 3.92, GRID, 0.7)

# Verified unsuccessful autonomous rollouts (LIBERO and StackCube).
pill(s, "未完成 / 失败", 8.78, 4.10, 1.48, RED, NAVY, 8.2)
add_image_cover(s, IMG_LIBERO_FULL, 8.78, 4.46, 1.62, 1.02, frame=True)
add_image_cover(s, IMG_STACK_200, 10.60, 4.46, 1.62, 1.02, frame=True)
add_text(s, "LIBERO · 180 步", 8.78, 5.62, 1.62, 0.17, 8.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_text(s, "自动运行未成功", 8.78, 5.84, 1.62, 0.16, 8.1, RED, True, FONT_CN, PP_ALIGN.CENTER)
add_text(s, "StackCube · 400 步", 10.60, 5.62, 1.62, 0.17, 8.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_text(s, "闭环未成功", 10.60, 5.84, 1.62, 0.16, 8.1, RED, True, FONT_CN, PP_ALIGN.CENTER)
add_text(s, "后续页面再拆：成功回放 / 未完成回放 / 诊断对照", 8.78, 6.35, 3.44, 0.28, 8.7, MUTED, False, FONT_CN, PP_ALIGN.CENTER)


# --- Slide 2: evidence first ---------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "OPEN WITH EVIDENCE", "先按结果分栏，再看原因", 2,
           "成功回放只放在左侧；未完成或失败回放只放在右侧，避免把 running 当成完成。")

# Left: one verified completed replay.  Keeping one large card makes the
# status easy to read before the audience sees the detailed diagnostics.
shape_rect(s, 0.72, 1.70, 5.72, 4.46, PANEL, 0.10, GRID, 0.7)
pill(s, "已完成回放", 0.98, 1.90, 1.22, GREEN, NAVY, 8.4)
add_text(s, "CALVIN · 单任务", 2.40, 1.94, 2.55, 0.20, 11.3, WHITE, True, FONT_CN)
video_card(s, IMG_SERVER_DRAWER_SEQ09, VID_SERVER_DRAWER_SEQ09,
           "CALVIN · 开抽屉", "sequence 09 · 判定成功",
           0.98, 2.28, 5.20, 2.80, GREEN, "6.8 s")
metric_card(s, "结果", "success", 1.00, 5.30, 1.70, GREEN, "sequence 09", 0.78)
metric_card(s, "步数", "63", 2.88, 5.30, 1.70, CYAN, "action count", 0.78)
metric_card(s, "来源", "server", 4.76, 5.30, 1.42, ORANGE, "CALVIN", 0.78)

# Right: the two autonomous rollouts that are not successful.  They are
# deliberately grouped under one status heading rather than interleaved with
# the completed CALVIN replay.
shape_rect(s, 6.70, 1.70, 5.92, 4.46, PANEL_ALT, 0.10, GRID, 0.7)
pill(s, "未完成 / 失败", 6.96, 1.90, 1.50, RED, NAVY, 8.2)
add_text(s, "自动运行结果", 8.66, 1.94, 2.72, 0.20, 11.3, WHITE, True, FONT_CN, PP_ALIGN.RIGHT)
video_card(s, IMG_LIBERO_FULL, VID_LIBERO_FULL,
           "LIBERO · 未成功", "180 步 · policy-only",
           6.96, 2.28, 2.66, 2.80, RED, "18.6 s")
video_card(s, IMG_STACK_200, VID_STACK,
           "StackCube · 未成功", "400 步 · 闭环",
           9.84, 2.28, 2.52, 2.80, RED, "20.1 s")
shape_rect(s, 6.96, 5.30, 5.40, 0.78, INK, 0.06, INK, 0.1)
add_text(s, "LIBERO：success = false", 7.16, 5.43, 2.35, 0.16, 9.0, WHITE, True, FONT_CN)
add_text(s, "StackCube：400 步仍未完成", 9.56, 5.43, 2.55, 0.16, 9.0, RED, True, FONT_CN, PP_ALIGN.RIGHT)
add_text(s, "这一页先把结果分开；后面再拆输入、起点和动作边界。",
         0.98, 6.40, 11.38, 0.18, 10.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "artifacts/libero_* + artifacts/server_media/ + runs/stackcube_v2_eval_20260910/", "证据按结果分栏 · 点击画面播放")


# --- Slide 3: mainline ------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "MAINLINE", "主线只有一条：先把单任务闭环跑通", 3,
           "环境、机制和 RL 都是支撑面；这轮先把输入对齐、单任务闭环和训练覆盖串成一条可复查的链。")
scope_cards = [
    (IMG_ROLLOUT_GRID, "01 · 对齐输入", "bridge / Schema30 / flattened state", "先保证每次 rollout 用的是同一份状态", CYAN),
    (IMG_RETARGET_ALL, "02 · 跑通单任务", "expert prefix / retarget / timing", "用少数对照定位长程闭环瓶颈", ORANGE),
    (IMG_STACK_200, "03 · 为扩展做准备", "StackCube / data / residual-SAC", "先把扩展所需的接口和回放链准备好", GREEN),
]
for i, (img, title, line1, line2, col) in enumerate(scope_cards):
    x = 0.72 + i * 4.06
    shape_rect(s, x, 1.72, 3.74, 4.72, PANEL_ALT if i == 1 else PANEL, 0.10, GRID, 0.7)
    add_image_cover(s, img, x + 0.14, 1.88, 3.46, 1.86, frame=True)
    pill(s, f"0{i+1}", x + 0.26, 2.02, 0.52, col, NAVY, 9.0)
    pill(s, "证据图", x + 2.20, 2.03, 1.08, PANEL_ALT, WHITE, 8.0)
    add_text(s, title, x + 0.20, 3.98, 3.26, 0.25, 15, col, True)
    add_text(s, line1, x + 0.20, 4.40, 3.24, 0.32, 12.2, WHITE, True)
    add_text(s, line2, x + 0.20, 4.88, 3.24, 0.48, 11.0, MUTED, False)
    add_line(s, x + 0.20, 5.60, x + 3.48, 5.60, GRID, 0.7)
    add_text(s, ["第一个 gate：输入和坐标一致", "第二个 gate：单任务 full loop", "第三个 gate：后续扩展的入口"][i],
             x + 0.20, 5.82, 3.20, 0.34, 10.8, WHITE, True)
shape_rect(s, 0.72, 6.58, 11.90, 0.36, PANEL, 0.08, GRID, 0.6)
add_text(s, "当前在第 2 步：单任务 full loop 加固中；第 3 步再进入多任务。", 0.98, 6.67, 11.35, 0.17, 12.2, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "artifacts/libero_* · artifacts/libero_retarget_20260910 · runs/stackcube_v2_eval_20260910", "主线：单任务 → 多任务")


# --- Slide 4: workstream timeline ----------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "MAINLINE MAP", "主线上的关键节点，最后都回到单任务闭环", 4,
           "三个节点按顺序推进：前两步服务当前单任务 gate，最后一步为后续扩展留入口。")
timeline_items = [
    ("09/08—10", "输入对齐", IMG_ROLLOUT_GRID, "bridge / state contract", "目标 ±30 mm 对照", CYAN),
    ("09/11—13", "单任务闭环", IMG_HANDOFF_EXPERT, "prefix / retarget / timing", "把长程变量拆开", ORANGE),
    ("next", "扩展准备", IMG_STACK_200, "data / replay / runner", "residual-SAC baseline", GREEN),
]
add_line(s, 1.00, 2.35, 12.25, 2.35, GRID, 1.6)
for i, (date, title, img, line1, line2, col) in enumerate(timeline_items):
    x = 0.72 + i * 4.06
    cx = x + 1.87
    shape_oval(s, cx - 0.12, 2.23, 0.24, 0.24, col)
    add_text(s, date, x + 0.40, 1.80, 2.94, 0.22, 10.5, col, True, FONT_LATIN, PP_ALIGN.CENTER)
    shape_rect(s, x, 2.74, 3.74, 2.72, PANEL_ALT if i % 2 else PANEL, 0.09, GRID, 0.7)
    add_image_cover(s, img, x + 0.14, 2.86, 3.46, 0.86, frame=True)
    add_text(s, title, x + 0.16, 3.90, 3.40, 0.22, 15.0, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
    add_text(s, line1, x + 0.16, 4.28, 3.40, 0.25, 11.8, col, True, FONT_CN, PP_ALIGN.CENTER)
    add_text(s, line2, x + 0.16, 4.68, 3.40, 0.22, 10.2, MUTED, False, FONT_CN, PP_ALIGN.CENTER)
    if i < len(timeline_items) - 1:
        draw_arrow(s, x + 3.78, 4.16, x + 4.00, 4.16, GRID, 0.9)
shape_rect(s, 0.72, 5.82, 11.90, 0.64, PANEL, 0.09, GRID, 0.7)
add_text(s, "主线", 1.02, 6.02, 0.90, 0.20, 12.5, CYAN, True)
add_text(s, "输入对齐  →  单任务 full loop  →  多 seed 加固  →  多任务扩展", 2.02, 5.98, 9.95, 0.24, 14.2, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "runs/libero_spatial_task0_bringup_20260908 · artifacts/libero_*", "09/08—09/13")


# --- Slide 5: model stack inventory --------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "APPENDIX / MODEL STACK", "支撑单任务闭环的模型栈", 5,
           "参数规模和 contract 只作为复查材料；主线判断仍回到真实 rollout 和可播放证据。")
shape_rect(s, 0.72, 1.72, 7.55, 4.84, PANEL, 0.10, GRID, 0.7)
add_text(s, "formal run · parameter inventory", 1.04, 2.02, 4.4, 0.24, 14, CYAN, True, FONT_LATIN)
add_text(s, "模块分组", 1.04, 2.42, 1.15, 0.18, 9.5, DIM, True)
add_text(s, "参数量", 6.48, 2.42, 1.25, 0.18, 9.5, DIM, True, FONT_CN, PP_ALIGN.RIGHT)
inventory = [
    ("Observation + G1/G2/G3", 38.75, "13.5M + 25.2M", CYAN),
    ("Intent + dynamics + P1/P2/P3", 38.57, "23.1M + 8.2M + 6.9M", ORANGE),
    ("Retained bottom / decoder", 42.30, "42.3M", GREEN),
    ("Controlled transition", 8.03, "8.0M", RED),
]
for i, (lab, val, raw, col) in enumerate(inventory):
    yy = 2.72 + i * 0.70
    add_text(s, lab, 1.04, yy, 2.65, 0.24, 11.2, WHITE, True)
    shape_rect(s, 3.66, yy + 0.04, 2.52, 0.25, GRID, 0.06)
    shape_rect(s, 3.66, yy + 0.04, 2.52 * val / 42.30, 0.25, col, 0.06)
    add_text(s, f"{val:.1f}M", 6.42, yy - 0.01, 1.18, 0.22, 12, col, True, FONT_LATIN, PP_ALIGN.RIGHT)
    add_text(s, raw, 3.66, yy + 0.34, 2.82, 0.16, 8.7, MUTED, False, FONT_LATIN)
for x, w, lab, val, sub, col in [
    (1.04, 2.20, "complete model", "168.4M", "trainable 152.0M", CYAN),
    (3.48, 2.20, "loss budget", "89.4%", "action group", ORANGE),
    (5.92, 1.92, "runtime", "2-pass", "proposal → refine", GREEN),
]:
    shape_rect(s, x, 5.54, w, 0.90, PANEL_ALT, 0.07, GRID, 0.6)
    add_text(s, lab.upper(), x + 0.10, 5.67, w - 0.20, 0.13, 7.2, DIM, True, FONT_LATIN)
    add_text(s, val, x + 0.10, 5.86, w - 0.20, 0.23, 15.5, col, True, FONT_LATIN)
    add_text(s, sub, x + 0.10, 6.22, w - 0.20, 0.12, 7.4, MUTED, False, FONT_CN)
shape_rect(s, 8.55, 1.72, 4.08, 4.84, PANEL_ALT, 0.10, GRID, 0.7)
add_image_cover(s, IMG_ROLLOUT_GRID, 8.87, 2.02, 3.44, 1.78, frame=True)
pill(s, "真实 rollout 仍是验收面", 9.15, 2.17, 2.88, CYAN, NAVY, 8.8)
add_text(s, "四个 contract", 8.87, 4.10, 2.20, 0.22, 14, ORANGE, True)
contracts = [
    ("24 × 18", "action field", CYAN),
    ("4 intervals", "future horizon", ORANGE),
    ("−8 / −4 / 0", "DINO / raw frame", GREEN),
    ("G → S → W → P", "typed evidence path", WHITE),
]
for i, (a, b, col) in enumerate(contracts):
    yy = 4.52 + i * 0.44
    shape_oval(s, 8.90, yy + 0.04, 0.11, 0.11, col)
    add_text(s, a, 9.14, yy, 1.18, 0.20, 11.2, col, True, FONT_LATIN)
    add_text(s, b, 10.42, yy, 1.78, 0.20, 10.1, MUTED, False, FONT_CN)
add_footer(s, "runs/libero_spatial_task0_bringup_20260908/formal · audit summary", "inventory ≠ performance")


# --- Slide 6: rollout loop ------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "CONTEXT / ROLLOUT LOOP", "一轮 rollout 里，模型反复做四件事：看 → 想 → 做 → 再看", 6,
           "下面是真实 rollout 的 12 帧采样；每轮只执行动作块的第 1 行，然后把新状态送回下一轮。")
# Put the thing first: a real sequence of observations from the workspace.
add_image_cover(s, IMG_ROLLOUT_GRID, 0.72, 1.72, 8.15, 4.62, frame=True)
pill(s, "真实 rollout · 12 帧", 0.94, 1.93, 1.62, CYAN, NAVY, 9.2)
shape_rect(s, 9.14, 1.72, 3.48, 4.62, PANEL, 0.10, GRID, 0.7)
add_text(s, "图里对应的动作", 9.47, 2.02, 2.75, 0.25, 15, WHITE, True)
loop_steps = [
    ("01", "看当前画面", "多帧画面 + 语言", CYAN),
    ("02", "找对象 / 目标", "对象事实与意图", ORANGE),
    ("03", "出一段动作", "预测下一段状态", GREEN),
    ("04", "执行一行，再看", "结果回到下一轮", CYAN),
]
for i, (n, title, body, col) in enumerate(loop_steps):
    yy = 2.52 + i * 0.73
    shape_oval(s, 9.48, yy, 0.38, 0.38, col)
    add_text(s, n, 9.48, yy + 0.10, 0.38, 0.16, 9.2, NAVY, True, FONT_LATIN, PP_ALIGN.CENTER)
    add_text(s, title, 10.00, yy + 0.01, 2.22, 0.19, 12.2, WHITE, True)
    add_text(s, body, 10.00, yy + 0.25, 2.22, 0.17, 9.2, MUTED, False)
    if i < len(loop_steps) - 1:
        add_line(s, 9.67, yy + 0.42, 9.67, yy + 0.67, GRID, 0.9)
pill(s, "完整回放见第 02 页", 9.47, 5.55, 2.48, PANEL_ALT, WHITE, 8.3)
shape_rect(s, 0.72, 6.55, 11.90, 0.37, PANEL_ALT, 0.08, GRID, 0.6)
add_text(s, "实现上对应：观测 / 对象事实 → 世界预测 → 动作块 → 执行后回读", 1.00, 6.65, 11.35, 0.17, 12.4, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "runs/libero_spatial_task0_bringup_20260908/rollout/videos_e8_180/contact_sheet.png", "图示 · 完整回放见第 02 页")


# --- Slide 4: evidence / identity ----------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "CONTEXT / EVIDENCE PATH", "三类材料，沿着同一条单任务闭环链复查", 7,
           "原始轨迹、一次运行和 trace 放在一起；文字只标注它们在主线中的位置。")
evidence_panels = [
    (IMG_ROLLOUT_GRID, "01 · 原始轨迹", "条件和对象先固定", "HDF5 / simulator · split 59 / 7 / 7", CYAN),
    (IMG_HANDOFF_COLD, "02 · 一次运行", "bridge 输出能直接回放", "DINOv2-base · 768 dim · 336×336", ORANGE),
    (IMG_HANDOFF_EXPERT, "03 · 复查证据", "结果与执行动作放在一起", "MP4 · episode JSON · executed action", GREEN),
]
for i, (img, tag, title, sub, col) in enumerate(evidence_panels):
    x = 0.72 + i * 4.06
    shape_rect(s, x, 1.72, 3.74, 3.74, PANEL_ALT if i == 1 else PANEL, 0.10, GRID, 0.7)
    add_image_cover(s, img, x + 0.14, 1.86, 3.46, 2.18, frame=True)
    pill(s, tag, x + 0.26, 2.02, 1.48, col, NAVY, 8.8)
    pill(s, "证据图", x + 2.18, 2.02, 1.18, PANEL_ALT, WHITE, 8.0)
    add_text(s, title, x + 0.20, 4.28, 3.30, 0.23, 14.5, WHITE, True)
    add_text(s, sub, x + 0.20, 4.62, 3.30, 0.32, 10.2, MUTED, False, FONT_CN)
    if i < len(evidence_panels) - 1:
        draw_arrow(s, x + 3.74, 3.56, x + 4.02, 3.56, GRID, 1.0)
shape_rect(s, 0.72, 5.78, 11.90, 0.78, PANEL, 0.09, GRID, 0.7)
add_text(s, "真正的工程增量", 1.00, 6.00, 1.55, 0.20, 12.5, CYAN, True)
add_text(s, "同一条实验身份从数据版本 → 配置 / hash → 视频 / trace 串起来；坐标、时序、起点和覆盖项都能分开复查。", 2.65, 5.96, 9.45, 0.32, 14, WHITE, True)
add_footer(s, "runs/stackcube_dataset_audit_20260910/REPORT.md · docs/research/", "图、视频、trace 对得上")


# --- Slide 5: coordinate contract ----------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "GATE 1 / COORDINATE CONTRACT", "先把坐标对齐，后面的闭环对照才可比较", 8,
           "v3 坐标合同把 flattened state 的时间前缀纳入索引，±30 mm 目标改动能在 simulator 中精确复现。")
shape_rect(s, 0.70, 1.72, 4.05, 4.65, PANEL, 0.1, GRID, 0.7)
add_text(s, "flattened init state", 1.02, 2.02, 2.9, 0.25, 15, CYAN, True, FONT_LATIN)
indices = [("time", "0", DIM), ("qpos free-joint", "9", ORANGE), ("bowl y", "11", GREEN)]
for i, (lab, val, col) in enumerate(indices):
    x = 1.02 + i * 1.10
    shape_rect(s, x, 2.70, 0.94, 0.78, PANEL_ALT, 0.07, GRID, 0.6)
    add_text(s, val, x, 2.82, 0.94, 0.27, 21, col, True, FONT_LATIN, PP_ALIGN.CENTER)
    add_text(s, lab, x + 0.03, 3.18, 0.88, 0.15, 8.2, MUTED, False, FONT_CN, PP_ALIGN.CENTER)
    if i < 2:
        add_line(s, x + 0.94, 3.09, x + 1.08, 3.09, GRID, 0.8)
add_text(s, "v3 约束", 1.02, 4.12, 1.2, 0.22, 13, ORANGE, True)
bullet(s, "应用值 = 请求值，误差 0.0 m", 1.05, 4.48, 3.25, size=13, accent=GREEN)
bullet(s, "5 次 zero-action warm-up 后再进入 policy", 1.05, 4.96, 3.25, size=13, accent=CYAN)
bullet(s, "当前统一采用 v3 索引，旧版结果不再混用", 1.05, 5.44, 3.25, size=13, accent=ORANGE)

image_card(s, IMG_ROLLOUT_GRID, "A · causal-prefix", "目标改动进入同一 state 索引", 5.10, 1.72, 3.45, 2.22, CYAN, "静态证据")
image_card(s, IMG_RETARGET_ALL, "B · terminal-suffix", "同一初始状态下做配对比较", 8.75, 1.72, 3.85, 2.22, ORANGE, "静态证据")
metric_card(s, "目标改动", "±30 mm", 5.12, 4.35, 1.75, ORANGE, "bowl y qpos", 0.94)
metric_card(s, "对象 XY motion", "≈ 5e−9 m", 7.05, 4.35, 1.95, GREEN, "4/4 rollouts stationary", 0.94)
metric_card(s, "matched pairs", "2 × 2", 9.20, 4.35, 1.55, GREEN, "A / B checkpoints", 0.94)
add_text(s, "这组配对固定坐标合同；后续对照沿用同一索引。", 5.12, 5.75, 6.2, 0.34, 16, WHITE, True)
add_footer(s, "artifacts/libero_causal_ab_20260910/README.md + v3_audit.json", "release trace · 配对证据")


# --- Slide 6: causal A/B --------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "GATE 1 / INPUT ALIGNMENT", "目标位置改动已经能在 simulator 中复现", 9,
           "这页验证输入和坐标合同，后面的对照统一沿用这套 state 索引。")
image_card(s, IMG_LIBERO_FULL, "A · causal-prefix", "EEF moves toward bowl; no contact", 0.70, 1.72, 5.85, 1.65, CYAN, "长回放截图")
image_card(s, IMG_LIBERO_FULL_E4, "B · terminal-suffix", "gripper closes earlier; no contact", 6.78, 1.72, 5.85, 1.65, ORANGE, "长回放截图")
metric_card(s, "A EEF displacement", "66.4 mm", 0.72, 3.77, 2.10, CYAN, "mean over 4", 0.94)
metric_card(s, "B EEF displacement", "60.7 mm", 2.98, 3.77, 2.10, ORANGE, "mean over 4", 0.94)
metric_card(s, "A y response", "1.01 mm", 5.24, 3.77, 2.10, CYAN, "1.69% of 60 mm", 0.94)
metric_card(s, "B y response", "0.76 mm", 7.50, 3.77, 2.10, ORANGE, "1.27% of 60 mm", 0.94)
metric_card(s, "probe", "配对", 9.76, 3.77, 1.55, GREEN, "matched checkpoints", 0.94)
shape_rect(s, 0.72, 5.12, 11.90, 1.18, PANEL, 0.09, GRID, 0.7)
add_text(s, "用途", 1.02, 5.43, 0.65, 0.22, 13, CYAN, True)
add_text(s, "确认目标改动确实进入同一条 state / action 索引；这组只做输入归因，单任务 full loop 另看长回放。", 1.80, 5.34, 10.25, 0.52, 15, WHITE, True)
add_footer(s, "v3_audit.json · 2 matched init-state pairs / checkpoint", "输入合同 · 配对证据")


# --- Slide 7: timing attribution -----------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "单任务诊断 / 夹爪时序", "只改夹爪时序，arm 轨迹保持一致", 10,
           "把 arm path 与 gripper command 分开看；下一轮再放回单任务长回放验证。")
image_card(s, IMG_RETARGET_BASE, "baseline", "normal bridge inference", 0.72, 1.75, 3.55, 2.45, CYAN, "release trace")
image_card(s, IMG_RETARGET_PLUS, "delayed-close", "first 12 positive gripper commands suppressed", 4.48, 1.75, 3.55, 2.45, ORANGE, "release trace")
shape_rect(s, 8.30, 1.75, 4.30, 2.45, PANEL, 0.10, GRID, 0.7)
add_text(s, "把两条信号分开看", 8.62, 2.03, 3.7, 0.25, 14, ORANGE, True)
add_text(s, "arm", 8.62, 2.54, 0.48, 0.18, 10.5, MUTED, True, FONT_LATIN)
add_text(s, "同一条", 11.66, 2.54, 0.62, 0.18, 9.2, GREEN, True, FONT_CN, PP_ALIGN.RIGHT)
for j in range(10):
    xx = 9.26 + j * 0.25
    shape_rect(s, xx, 2.54, 0.18, 0.18, GREEN if j in (2, 3, 4, 5, 6, 7) else GRID, 0.04)
add_text(s, "gripper", 8.62, 3.19, 0.70, 0.18, 10.5, MUTED, True, FONT_LATIN)
add_text(s, "晚 12 步", 11.44, 3.19, 0.84, 0.18, 9.2, ORANGE, True, FONT_CN, PP_ALIGN.RIGHT)
for j in range(10):
    xx = 9.26 + j * 0.25
    shape_rect(s, xx, 3.19, 0.18, 0.18, ORANGE if j in (6, 7, 8, 9) else GRID, 0.04)
add_text(s, "arm Δ RMS = 0 · gripper Δ = 0.441", 8.62, 3.74, 3.55, 0.20, 10.2, WHITE, True, FONT_LATIN)
metric_card(s, "delay", "12 步", 0.72, 4.72, 2.15, ORANGE, "gripper timing", 0.92)
metric_card(s, "arm action", "exact copy", 3.05, 4.72, 2.15, GREEN, "Δ RMS = 0", 0.92)
metric_card(s, "对照", "配对", 5.38, 4.72, 2.15, CYAN, "deployment-only", 0.92)
shape_rect(s, 7.72, 4.72, 4.90, 1.55, PANEL_ALT, 0.09, GRID, 0.7)
add_text(s, "用途", 8.05, 5.04, 0.7, 0.22, 13, CYAN, True)
add_text(s, "夹爪命令会变，但 arm 控制轨迹保持一致；后续把时序变量单独放进单任务 full-loop 对照。", 8.05, 5.40, 4.2, 0.52, 14, WHITE, True)
add_footer(s, "artifacts/libero_timing_20260911/timing_probe_v1.json", "deployment-only · timing pair")


# --- Slide 8: expert handoff ---------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "DIAGNOSTIC / CONTROLLED PAIR", "诊断对照：冷启动与 expert prefix", 8,
           "同一任务、同一策略，只改 reset 后的 36 步；两段结果用于定位起点影响，不是同类成功回放。")
video_card(s, IMG_HANDOFF_COLD, VID_HANDOFF_COLD, "未成功 · policy-only", "reset 起点 · 0 / 200", 0.70, 1.72, 5.85, 2.35, RED, "20.6 s")
video_card(s, IMG_HANDOFF_EXPERT, VID_HANDOFF_EXPERT, "成功 · expert prefix 36 步", "policy 接管后 · step 79 完成（诊断）", 6.78, 1.72, 5.85, 2.35, GREEN, "20.6 s")
add_line(s, 1.05, 4.70, 11.95, 4.70, GRID, 1.6)
for x, label, col in [(1.15, "reset", DIM), (3.10, "expert 36", ORANGE), (5.55, "handoff", CYAN), (8.50, "policy 164", GREEN), (11.20, "step 200", WHITE)]:
    shape_oval(s, x, 4.58, 0.24, 0.24, col)
    add_text(s, label, x - 0.35, 5.02, 0.95, 0.19, 9.2, col, True, FONT_CN, PP_ALIGN.CENTER)
metric_card(s, "cold start", "false", 0.75, 5.55, 2.35, RED, "policy-only", 0.92)
metric_card(s, "handoff", "true", 3.28, 5.55, 2.35, GREEN, "step 79 · diagnostic", 0.92)
metric_card(s, "prefix", "36 步", 5.81, 5.55, 2.35, ORANGE, "expert actions", 0.92)
metric_card(s, "policy", "164 步", 8.34, 5.55, 2.35, CYAN, "after handoff", 0.92)
add_text(s, "用途：把 reset 后的前 36 步变成可控初始化变量，再检查 policy 接管后的单任务长程闭环。", 0.78, 6.65, 11.65, 0.26, 13.5, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2.json", "内嵌视频 · 点击画面播放")


# --- Slide 9: retarget / release-centered replay -------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "GATE 2 / ACTION BOUNDARY", "单任务动作边界：release / retarget 做成三组对照", 9,
           "同一初始状态下只改一个条件；把动作差异留在可回放、可比较的变量里。")
shape_rect(s, 0.72, 1.72, 11.90, 1.70, PANEL, 0.10, GRID, 0.7)
pill(s, "6 组 paired replay · release row", 1.08, 1.94, 2.58, ORANGE, NAVY, 8.8)
add_text(s, "同一桌面、同一任务、不同 release / retarget 条件", 7.20, 1.98, 4.62, 0.18, 10.8, WHITE, True, FONT_CN, PP_ALIGN.RIGHT)
add_line(s, 4.08, 2.38, 11.70, 2.38, GRID, 1.0)
add_line(s, 4.08, 2.76, 11.70, 2.76, GRID, 1.0)
add_line(s, 4.08, 3.14, 11.70, 3.14, GRID, 1.0)
add_line(s, 8.92, 2.20, 8.92, 3.24, GRID, 1.0)
for yy, lab, val, col in [(2.30, "baseline", 0.2265, CYAN), (2.68, "+3 cm", -0.4698, ORANGE), (3.06, "−3 cm", -1.0, GREEN)]:
    add_text(s, lab, 1.08, yy - 0.08, 1.20, 0.18, 10.0, WHITE, True, FONT_LATIN)
    end = 8.92 + val * 4.84
    shape_rect(s, min(8.92, end), yy, abs(end - 8.92), 0.16, col, 0.04)
    shape_oval(s, end - 0.06, yy - 0.02, 0.12, 0.20, col)
    sign = "+" if val > 0 else ""
    add_text(s, f"{sign}{val:.4f}", 11.74, yy - 0.08, 0.70, 0.18, 9.4, col, True, FONT_LATIN, PP_ALIGN.RIGHT)
add_text(s, "release row 85", 4.10, 3.24, 1.20, 0.14, 8.0, DIM, False, FONT_LATIN)
add_text(s, "same init state", 9.55, 3.24, 1.30, 0.14, 8.0, DIM, False, FONT_LATIN)
retarget_cards = [
    (IMG_RETARGET_BASE, "baseline", "原始 release 条件", CYAN),
    (IMG_RETARGET_PLUS, "+3 cm", "目标 retarget · plus", ORANGE),
    (IMG_RETARGET_MINUS, "−3 cm", "目标 retarget · minus", GREEN),
]
for i, (img, lab, sub, col) in enumerate(retarget_cards):
    x = 0.72 + i * 4.06
    shape_rect(s, x, 3.70, 3.74, 2.52, PANEL_ALT if i == 1 else PANEL, 0.10, GRID, 0.7)
    add_image_cover(s, img, x + 0.14, 3.86, 3.46, 0.98, frame=True)
    pill(s, lab, x + 0.26, 4.98, 0.90, col, NAVY, 8.7)
    pill(s, "数值证据", x + 2.15, 4.98, 1.22, PANEL_ALT, WHITE, 8.0)
    add_text(s, sub, x + 0.20, 5.43, 3.20, 0.20, 11.0, WHITE, True)
    add_text(s, ["release row 85: +0.2265", "release row 85: −0.4698", "下一行: −1.0"][i],
             x + 0.20, 5.78, 3.20, 0.18, 9.8, col, True, FONT_LATIN)
shape_rect(s, 0.72, 6.42, 11.90, 0.32, PANEL, 0.08, GRID, 0.6)
add_text(s, "这组材料把单任务的 release / retarget 边界固定下来，下一轮沿同一初始状态继续拉长 horizon。", 0.98, 6.49, 11.35, 0.18, 11.5, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "artifacts/libero_retarget_20260910/closed_loop_r3_v4/", "paired replay · release trace")


# --- Slide 10: formal training curve -------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "TRAINING SUPPORT / 4 EPOCHS", "训练支撑：4 个 epoch 的误差轨迹", 10,
           "这页用于选择下一轮单任务重跑的起点；事件头的细节另放诊断材料。")
shape_rect(s, 0.72, 1.72, 7.55, 4.84, PANEL, 0.10, GRID, 0.7)
add_text(s, "validation RMSE · physical units", 1.04, 2.02, 4.2, 0.22, 14, CYAN, True, FONT_LATIN)
plot_x, plot_y, plot_w, plot_h = 1.20, 2.55, 6.20, 3.05
add_line(s, plot_x, plot_y, plot_x, plot_y + plot_h, GRID, 0.9)
add_line(s, plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, GRID, 0.9)
for tick, lab in [(0.0, "0"), (0.7, "0.7"), (1.4, "1.4")]:
    yy = plot_y + plot_h - tick / 1.4 * plot_h
    add_line(s, plot_x, yy, plot_x + plot_w, yy, GRID, 0.5)
    add_text(s, lab, 0.84, yy - 0.09, 0.27, 0.16, 8.5, DIM, False, FONT_LATIN, PP_ALIGN.RIGHT)
formal_series = [
    ("full", [0.6463276, 0.5201054, 0.3737026, 0.2979138], CYAN),
    ("arm", [0.4556660, 0.3133159, 0.2119910, 0.1739029], GREEN),
    ("gripper", [1.2955254, 1.1421760, 0.8413885, 0.6631858], ORANGE),
]
for name, vals, col in formal_series:
    pts = []
    for i, v in enumerate(vals):
        px = plot_x + plot_w * i / 3.0
        py = plot_y + plot_h - v / 1.4 * plot_h
        pts.append((px, py))
    for a, b in zip(pts, pts[1:]):
        add_line(s, a[0], a[1], b[0], b[1], col, 2.2)
    for px, py in pts:
        shape_oval(s, px - 0.055, py - 0.055, 0.11, 0.11, col)
    add_text(s, name, pts[-1][0] + 0.10, pts[-1][1] - 0.10, 0.68, 0.18, 9.5, col, True, FONT_LATIN)
for i, epoch in enumerate(["E1", "E2", "E3", "E4"]):
    px = plot_x + plot_w * i / 3.0
    add_text(s, epoch, px - 0.18, 5.78, 0.36, 0.18, 9.2, MUTED, True, FONT_LATIN, PP_ALIGN.CENTER)
for i, (lab, col) in enumerate([("full", CYAN), ("arm", GREEN), ("gripper", ORANGE)]):
    shape_oval(s, 1.04 + i * 1.12, 6.12, 0.10, 0.10, col)
    add_text(s, lab, 1.20 + i * 1.12, 6.06, 0.78, 0.18, 9.5, MUTED, False, FONT_LATIN)
shape_rect(s, 8.55, 1.72, 4.08, 4.84, PANEL_ALT, 0.10, GRID, 0.7)
add_text(s, "四个读数", 8.88, 2.02, 2.0, 0.23, 14, ORANGE, True)
metric_card(s, "full RMSE", "0.646 → 0.298", 8.88, 2.48, 3.42, CYAN, "epoch 1 → 4", 0.78)
metric_card(s, "arm RMSE", "0.456 → 0.174", 8.88, 3.42, 3.42, GREEN, "epoch 1 → 4", 0.78)
metric_card(s, "gripper RMSE", "1.296 → 0.663", 8.88, 4.36, 3.42, ORANGE, "epoch 1 → 4", 0.78)
metric_card(s, "event ratio", "33.2 → 28.2", 8.88, 5.30, 3.42, ORANGE, "epoch 1 → 4", 0.78)
add_footer(s, "runs/libero_spatial_task0_bringup_20260908/formal · audit summary", "4 epochs · peak ≈ 21.2 GiB")


# --- Slide 11: training health -------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "TRAINING SUPPORT / PROBES", "训练支撑：内部 probe 已能读出关键路径", 14,
           "结构探针、梯度和事件头用于验收训练路径，结果回到单任务闭环。")
shape_rect(s, 0.72, 1.72, 4.36, 4.84, PANEL, 0.10, GRID, 0.7)
add_image_cover(s, IMG_ROLLOUT_GRID, 0.94, 2.02, 3.92, 2.16, frame=True)
pill(s, "现场验收面", 1.16, 2.18, 1.28, CYAN, NAVY, 8.8)
add_text(s, "内部 probe 的读法", 1.02, 4.54, 2.50, 0.22, 14, CYAN, True)
bullet(s, "execution progress：0 → 0.497", 1.04, 4.96, 3.62, size=11.5, accent=GREEN)
bullet(s, "value correlation：0.379 → 0.544", 1.04, 5.36, 3.62, size=11.5, accent=ORANGE)
bullet(s, "pairwise accuracy：0.798 → 0.832", 1.04, 5.76, 3.62, size=11.5, accent=CYAN)
shape_rect(s, 5.34, 1.72, 7.28, 4.84, PANEL_ALT, 0.10, GRID, 0.7)
add_text(s, "训练健康度快照", 5.68, 2.02, 2.80, 0.23, 14, ORANGE, True)
health_cards = [
    ("proposal zero gain", "0.000", "neutral ablation", GREEN),
    ("motion F1", "1.000", "head-level event", GREEN),
    ("flow JEPA floor", "2.3e−6", "aligned G2 input", CYAN),
    ("gradient preclip max", "5.59", "spike audit", ORANGE),
    ("value corr", "0.544", "formal tail", CYAN),
    ("event ratio", "28.2×", "formal epoch 4", ORANGE),
]
for i, (lab, val, sub, col) in enumerate(health_cards):
    x = 5.68 + (i % 2) * 3.48
    y = 2.48 + (i // 2) * 1.08
    metric_card(s, lab, val, x, y, 3.12, col, sub, 0.88)
shape_rect(s, 5.68, 5.86, 6.48, 0.48, PANEL, 0.08, GRID, 0.6)
add_text(s, "用途：确认训练路径可读，再把事件覆盖和起点覆盖带回单任务 full-loop。", 5.92, 5.96, 5.98, 0.28, 10.7, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "runs/libero_spatial_task0_bringup_20260908/formal · release-first audit", "诊断工具 · 非成功率")


# --- Slide 12: release-first training ------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "训练快照 / 释放事件", "把释放条件纳入训练快照", 12,
           "1 epoch / 120 batches 的快照；它给下一轮单任务闭环提供统一的可复查起点。")
shape_rect(s, 0.72, 1.72, 6.00, 4.85, PANEL, 0.10, GRID, 0.7)
add_text(s, "source-native validation · r3 → r4", 1.02, 2.04, 4.4, 0.23, 14, CYAN, True, FONT_LATIN)
rows = [
    ("action RMSE", 0.213688, 0.200630, CYAN),
    ("arm RMSE", 0.131133, 0.122967, GREEN),
    ("gripper RMSE", 0.465255, 0.437082, ORANGE),
]
for i, (lab, old, new, col) in enumerate(rows):
    yy = 2.72 + i * 0.78
    add_text(s, lab, 1.02, yy, 1.55, 0.2, 10.5, MUTED, False, FONT_CN)
    shape_rect(s, 2.64, yy + 0.02, 2.25, 0.18, GRID, 0.05)
    shape_rect(s, 2.64, yy + 0.02, 2.25 * old / 0.50, 0.18, DIM, 0.05)
    shape_rect(s, 2.64, yy + 0.27, 2.25, 0.18, GRID, 0.05)
    shape_rect(s, 2.64, yy + 0.27, 2.25 * new / 0.50, 0.18, col, 0.05)
    add_text(s, f"{old:.3f}", 5.03, yy - 0.02, 0.62, 0.18, 9.3, DIM, True, FONT_LATIN, PP_ALIGN.RIGHT)
    add_text(s, f"{new:.3f}", 5.03, yy + 0.23, 0.62, 0.18, 9.3, col, True, FONT_LATIN, PP_ALIGN.RIGHT)
    add_text(s, "r3", 2.18, yy + 0.02, 0.35, 0.16, 8.2, DIM, False, FONT_LATIN, PP_ALIGN.RIGHT)
    add_text(s, "r4", 2.18, yy + 0.27, 0.35, 0.16, 8.2, col, True, FONT_LATIN, PP_ALIGN.RIGHT)
add_text(s, "release row 85", 1.02, 5.38, 1.35, 0.2, 12, ORANGE, True, FONT_LATIN)
add_text(s, "+0.2265  →  −0.4698", 2.38, 5.32, 2.82, 0.28, 20.5, GREEN, True, FONT_LATIN)
add_text(s, "next row: −1.0", 5.48, 5.40, 1.12, 0.18, 9.7, MUTED, False, FONT_LATIN)
shape_rect(s, 7.02, 1.72, 5.60, 4.85, PANEL_ALT, 0.10, GRID, 0.7)
add_text(s, "r4 validation snapshot", 7.34, 2.04, 3.5, 0.23, 14, ORANGE, True, FONT_CN)
metric_card(s, "proposal RMSE", f"{num(release_val.get('validation_proposal_primary_rmse_physical', 0.20063)):.3f}", 7.34, 2.54, 2.25, GREEN, "physical", 0.92)
metric_card(s, "tail RMSE", f"{num(release_val.get('validation_tail_rmse_physical', 0.197623)):.3f}", 9.82, 2.54, 2.25, CYAN, "physical", 0.92)
metric_card(s, "event F1", f"{num(release_val.get('validation_p2_intervention_primary_decoded_gripper_event_f1', 0.118721)):.3f}", 7.34, 3.70, 2.25, ORANGE, "precision 0.063", 0.92)
metric_card(s, "event recall", f"{num(release_val.get('validation_p2_intervention_primary_decoded_gripper_event_recall', 1.0)):.1f}", 9.82, 3.70, 2.25, RED, "pred/target = 412/26", 0.92)
add_text(s, "探针与 loss ledger 已能复查；下一轮把 release 事件与 early-state 覆盖带回 full loop。", 7.34, 5.24, 4.82, 0.68, 13.2, WHITE, True)
add_footer(s, "artifacts/libero_release_first_20260913/r4_releasefirst/audit.json + summary.md", "epoch 1 · step 120")


# --- Slide 10: StackCube / RL --------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "PARALLEL PREP / STACKCUBE", "StackCube：接入残差 Soft Actor-Critic（residual-SAC）基线", 13,
           "底层策略先给出动作，residual-SAC 学 Δa 修正；当前完成的是环境、数据、replay 和 runner 接入。")
for i, (img, label, col) in enumerate([(IMG_STACK_0, "step 000", CYAN), (IMG_STACK_100, "step 100", ORANGE), (IMG_STACK_200, "step 200", GREEN), (IMG_STACK_400, "step 400", RED)]):
    x = 0.72 + i * 1.22
    add_image_cover(s, img, x, 1.75, 1.08, 1.68, frame=True)
    pill(s, label, x + 0.05, 3.51, 0.98, col, NAVY)
pill(s, "完整回放见第 18 页", 5.82, 1.95, 1.55, PANEL_ALT, WHITE, 8.3)
shape_rect(s, 0.72, 4.05, 4.70, 1.45, PANEL, 0.10, GRID, 0.7)
add_text(s, "从数据到 replay", 1.00, 4.27, 2.20, 0.22, 13.5, CYAN, True)
chain = [("59 / 7 / 7", "split", CYAN), ("24 × 18", "action", ORANGE), ("400", "steps", GREEN)]
for i, (val, lab, col) in enumerate(chain):
    xx = 0.98 + i * 1.36
    shape_rect(s, xx, 4.67, 1.12, 0.48, PANEL_ALT, 0.07, GRID, 0.6)
    add_text(s, val, xx + 0.04, 4.76, 1.04, 0.18, 12.2, col, True, FONT_LATIN, PP_ALIGN.CENTER)
    add_text(s, lab, xx + 0.04, 4.98, 1.04, 0.12, 8.2, MUTED, False, FONT_CN, PP_ALIGN.CENTER)
    if i < len(chain) - 1:
        draw_arrow(s, xx + 1.13, 4.91, xx + 1.31, 4.91, GRID, 0.8)
add_text(s, "policy latency median 3.13 s · E6 专家前缀第 44 步抓到红块", 1.00, 5.26, 4.08, 0.16, 8.9, MUTED, False, FONT_CN)
shape_rect(s, 5.70, 2.45, 6.90, 3.65, PANEL, 0.10, GRID, 0.7)
add_text(s, "接入面", 6.03, 2.75, 1.1, 0.22, 14, CYAN, True)
metric_card(s, "episodes", "59 / 7 / 7", 6.03, 3.20, 2.00, CYAN, "train / val / test", 0.90)
metric_card(s, "native RMSE", "0.188", 8.22, 3.20, 2.00, ORANGE, "E3 · source-native", 0.90)
metric_card(s, "replay", "400 步", 10.41, 3.20, 1.72, GREEN, "closed loop", 0.90)
bullet(s, "最小 TCP→cube 距离 83.3 mm", 6.05, 4.53, 5.85, size=13, accent=WHITE)
bullet(s, "最小 cube→goal 距离 250.4 mm", 6.05, 4.98, 5.85, size=13, accent=WHITE)
bullet(s, "a = a_base + Δa：残差只修正底层策略动作", 6.05, 5.43, 5.85, size=13, accent=GREEN)
add_text(s, "下一步：baseline → adapter → 多 seed", 0.76, 5.92, 4.90, 0.28, 13.0, WHITE, True, FONT_CN)
add_footer(s, "runs/stackcube_v2_eval_20260910/ + runs/stackcube_dataset_audit_20260910/REPORT.md", "replay video 可点击")


# --- Slide 11: data coverage gaps ----------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "GATE 2 / TRAINING COVERAGE", "为单任务闭环补训练覆盖：首行要和部署顺序对齐", 17,
           "这页把监督序列与部署首行对齐，作为单任务重跑前的检查。")
# Keep a real rollout visible so the audit numbers have a physical reference.
shape_rect(s, 0.72, 1.75, 7.12, 4.80, PANEL, 0.10, GRID, 0.7)
add_text(s, "先看部署端的 400 步轨迹", 1.04, 2.05, 4.1, 0.25, 15, CYAN, True)
for i, (img, lab, col) in enumerate([
    (IMG_STACK_0, "0", CYAN), (IMG_STACK_100, "100", ORANGE),
    (IMG_STACK_200, "200", GREEN), (IMG_STACK_400, "400", RED)
]):
    xx = 1.04 + i * 1.54
    add_image_cover(s, img, xx, 2.44, 1.34, 1.38, frame=True)
    pill(s, f"step {lab}", xx + 0.08, 3.92, 1.18, col, NAVY, 8.4)
add_text(s, "部署会走一条完整时间轴；训练端要先在相同位置看见 release。", 1.04, 4.40, 6.1, 0.22, 11.5, MUTED)
add_text(s, "first execution row", 1.04, 4.90, 2.25, 0.2, 12.5, WHITE, True, FONT_LATIN)
shape_rect(s, 1.04, 5.26, 5.98, 0.34, GRID, 0.08)
for j in range(24):
    xx = 1.08 + j * 0.245
    col = ORANGE if j == 23 else (PANEL_ALT if j < 8 else GRID)
    shape_rect(s, xx, 5.30, 0.18, 0.26, col, 0.03)
add_text(s, "release 在 24 行目标里", 1.04, 5.72, 2.35, 0.18, 10.5, ORANGE, True, FONT_CN)
add_text(s, "但前 8 行：0 / 59", 4.56, 5.72, 2.44, 0.18, 10.5, RED, True, FONT_CN, PP_ALIGN.RIGHT)
shape_rect(s, 8.12, 1.75, 4.50, 4.80, PANEL_ALT, 0.10, GRID, 0.7)
add_text(s, "为什么看这两个数字", 8.46, 2.05, 3.25, 0.25, 15, ORANGE, True)
add_text(s, "部署首行 = release；前 8 行要有同类事件", 8.46, 2.34, 3.60, 0.18, 10.0, WHITE, True, FONT_CN)
add_text(s, "第 1 执行行", 8.46, 2.58, 1.65, 0.18, 11, MUTED, True, FONT_CN)
add_text(s, "0 / 59", 10.18, 2.49, 1.96, 0.35, 27, RED, True, FONT_LATIN, PP_ALIGN.RIGHT)
add_text(s, "前 8 行", 8.46, 3.17, 1.65, 0.18, 11, MUTED, True, FONT_CN)
add_text(s, "0 / 59", 10.18, 3.08, 1.96, 0.35, 27, ORANGE, True, FONT_LATIN, PP_ALIGN.RIGHT)
add_line(s, 8.46, 3.70, 12.12, 3.70, GRID, 0.8)
add_text(s, "B4 event quota", 8.46, 3.96, 2.45, 0.22, 14, CYAN, True, FONT_LATIN)
for j, (lab, col) in enumerate([("U", CYAN), ("U", CYAN), ("E", RED), ("E", RED)]):
    xx = 8.46 + j * 0.66
    shape_rect(s, xx, 4.38, 0.52, 0.42, col if col != RED else PANEL, 0.06, col, 0.8)
    add_text(s, lab, xx, 4.51, 0.52, 0.16, 11, NAVY if col == CYAN else RED, True, FONT_LATIN, PP_ALIGN.CENTER)
add_text(s, "round(4 × 0.125) = 0", 8.46, 5.04, 3.55, 0.25, 17, ORANGE, True, FONT_LATIN)
add_text(s, "先改跨 batch 配额，再重跑单任务。", 8.46, 5.54, 3.55, 0.34, 13.2, WHITE, True, FONT_CN)
add_footer(s, "runs/stackcube_dataset_audit_20260910/REPORT.md", "数据覆盖 · 下一轮入口")


# --- Slide 15: media wall -------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "UNFINISHED / DIAGNOSTIC REPLAYS", "未完成回放：问题出现在哪里", 18,
           "以下片段只用于定位自动运行、任务接力和接近段的边界，不作为成功展示。")
video_card(s, IMG_LIBERO_FULL_E4, VID_LIBERO_FULL_E4, "1 · LIBERO · 未成功", "180 步 · policy-only", 0.72, 1.72, 5.78, 2.18, RED, "18.6 s")
video_card(s, IMG_SERVER_MULTI_SEQ05_360, VID_SERVER_MULTI_SEQ05_360, "2 · CALVIN seq05 · 未完成", "task 2/5 · step 350/360 · running", 6.78, 1.72, 5.78, 2.18, ORANGE, "37.1 s")
video_card(s, IMG_SERVER_DRAWER_TARGETED_163, VID_SERVER_DRAWER_TARGETED_163, "3 · CALVIN 开抽屉 · 状态未核定", "server targeted · 163 步 · 诊断片段", 0.72, 4.20, 5.78, 2.18, ORANGE, "15.2 s")
video_card(s, IMG_STACK_200, VID_STACK_APPROACH, "4 · StackCube · 未完成", "400 步 · approach / no stack", 6.78, 4.20, 5.78, 2.18, RED, "20.1 s")
shape_rect(s, 0.72, 6.55, 11.90, 0.28, PANEL_ALT, 0.06, GRID, 0.6)
add_text(s, "状态明确的成功回放放在 CALVIN 单任务页；这里保留未完成和状态未核定的片段。", 0.98, 6.61, 11.35, 0.14, 9.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "runs/libero_* · artifacts/server_media/ · runs/stackcube_v2_eval_20260910/", "未完成 / 诊断 · 点击画面播放")


# --- Slide 16: CALVIN media ------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "CROSS-ENVIRONMENT / CALVIN", "CALVIN：已完成回放与未完成回放分开", 19,
           "上行只放单任务成功；下行放多任务或失败片段，避免把 RUNNING 当成完成。", ORANGE)

# Completed single-task replays.
shape_rect(s, 0.72, 1.66, 11.90, 2.42, PANEL, 0.10, GRID, 0.7)
pill(s, "单任务 · 已完成", 0.98, 1.82, 1.52, GREEN, NAVY, 8.3)
add_text(s, "success 字段为 true 的回放", 9.18, 1.87, 3.05, 0.18, 9.5, MUTED, True, FONT_CN, PP_ALIGN.RIGHT)
video_card(s, IMG_SERVER_DRAWER_E4_245, VID_SERVER_DRAWER_E4_245,
           "1 · 开抽屉 · 成功", "CALVIN server · 245 步",
           0.98, 2.14, 5.54, 1.78, GREEN, "13.8 s")
video_card(s, IMG_CALVIN_PUSH_POSTER, VID_CALVIN_PUSH,
           "2 · 推蓝色方块 · 成功", "本地归档 · 73 步",
           6.82, 2.14, 5.54, 1.78, GREEN, "6.7 s")

# Unfinished or unsuccessful replays.
shape_rect(s, 0.72, 4.22, 11.90, 2.42, PANEL_ALT, 0.10, GRID, 0.7)
pill(s, "未完成 / 失败", 0.98, 4.38, 1.42, RED, NAVY, 8.3)
add_text(s, "多任务接力与失败对照", 9.42, 4.43, 2.80, 0.18, 9.5, MUTED, True, FONT_CN, PP_ALIGN.RIGHT)
video_card(s, IMG_SERVER_MULTI_SEQ07_360, VID_SERVER_MULTI_SEQ07_360,
           "3 · 多任务 seq07 · 未完成", "task 2/5 · step 350/360 · running",
           0.98, 4.70, 5.54, 1.78, ORANGE, "37.1 s")
video_card(s, IMG_CALVIN_PUSH_FAILURE, VID_CALVIN_PUSH_FAILURE,
           "4 · 推蓝色方块 · 未成功", "alt position · 360 步 · RUNNING",
           6.82, 4.70, 5.54, 1.78, RED, "30.6 s")
add_text(s, "同一页只做结果分组：上行已完成，下行未完成或失败。", 0.98, 6.78, 11.38, 0.16, 9.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "artifacts/server_media/ · new_logs/current/calvin/closed_loop_videos/", "内嵌视频 · 点击画面播放")


# --- Slide 17: evidence boundaries ---------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "WHAT WE CAN SAY", "当前单任务证据，先收成三句话", 16,
           "三句话分别对应输入、归因和起点；后续实验沿着同一条单任务 gate 继续。")
cards = [
    (IMG_LIBERO_FULL, "01 · 输入", "±30 mm 目标能复现；对象保持静止", CYAN),
    (IMG_HANDOFF_COLD, "02 · 归因", "晚关 12 步；arm path 保持一致", ORANGE),
    (IMG_HANDOFF_EXPERT, "03 · 起点", "expert prefix 36 步；step 79 完成", GREEN),
]
for i, (img, label, sub, col) in enumerate(cards):
    xx = 0.72 + i * 4.06
    image_card(s, img, label, sub, xx, 1.72, 3.74, 3.38, col, "结论")
shape_rect(s, 0.72, 5.46, 11.90, 0.96, PANEL, 0.09, GRID, 0.7)
add_text(s, "下一道 gate", 1.02, 5.72, 1.20, 0.20, 12.5, CYAN, True)
add_text(s, "单任务 full loop  →  多 seed 加固  →  再进入多任务；StackCube residual-SAC 作为并行准备线。", 2.35, 5.68, 9.75, 0.28, 14.0, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "current architecture contract · recent probes", "三句话 · 同一条 gate")


# --- Slide 17: decision board ----------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "DECISION BOARD", "决策顺序：先单任务跑通，再扩多任务", 20,
           "主线只推进一件事；每一级有自己的 gate，按顺序推进。")
decision_cards = [
    (IMG_HANDOFF_EXPERT, "01", "单任务 full loop", "reset → approach → grasp\n→ release / place", "一任务 · 多 seed · 完整回放", CYAN),
    (IMG_RETARGET_ALL, "02", "单任务加固", "early-state / release\nrecovery / event coverage", "事件覆盖 · 长程稳定", ORANGE),
    (IMG_ROLLOUT_GRID, "03", "多任务扩展", "task mix / transfer\n只在前两级通过后进入", "共享输入合同 · 再看泛化", GREEN),
]
for i, (img, n, title, body, gate, col) in enumerate(decision_cards):
    x = 0.72 + i * 4.06
    if i < len(decision_cards) - 1:
        draw_arrow(s, x + 3.78, 3.40, x + 4.00, 3.40, GRID, 1.0)
    shape_rect(s, x, 1.78, 3.74, 3.62, PANEL_ALT if i == 1 else PANEL, 0.10, GRID, 0.7)
    add_image_cover(s, img, x + 0.14, 1.94, 3.46, 1.30, frame=True)
    pill(s, n, x + 0.25, 2.08, 0.48, col, NAVY, 9.2)
    pill(s, "证据图", x + 2.34, 2.08, 1.05, PANEL_ALT, WHITE, 8.0)
    add_text(s, title, x + 0.20, 3.52, 3.20, 0.24, 15, col, True)
    add_text(s, body, x + 0.20, 3.92, 3.20, 0.56, 12.2, WHITE, True, FONT_CN)
    add_text(s, f"gate · {gate}", x + 0.20, 4.88, 3.20, 0.22, 10.2, MUTED, False, FONT_CN)
shape_rect(s, 0.72, 5.72, 11.90, 0.72, PANEL, 0.09, GRID, 0.7)
add_text(s, "并行准备线", 1.02, 5.96, 1.35, 0.20, 12.5, RED, True)
add_text(s, "StackCube · residual Soft Actor-Critic：先过 baseline，再训 residual adapter，单独验证。", 2.55, 5.91, 9.60, 0.28, 13.2, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "workspace artifacts and runbooks", "主线收束")


# --- Slide 22: evidence index --------------------------------------------
s = prs.slides.add_slide(BLANK)
add_header(s, "APPENDIX / EVIDENCE INDEX", "每个判断，都能回到一组素材", 22,
           "这一页只做索引：视频、contact sheet、JSON trace 和 audit report 都留在工作区，方便会后复查。")
index_rows = [
    ("LIBERO bridge / formal", "runs/libero_spatial_task0_bringup_20260908/", IMG_ROLLOUT_GRID, CYAN),
    ("Causal A/B + timing", "artifacts/libero_causal_ab_20260910/\nartifacts/libero_timing_20260911/", IMG_HANDOFF_COLD, ORANGE),
    ("Retarget + handoff", "artifacts/libero_retarget_20260910/\nartifacts/libero_release_first_20260913/", IMG_RETARGET_ALL, GREEN),
    ("StackCube / RL audit", "runs/stackcube_v2_eval_20260910/\nruns/stackcube_dataset_audit_20260910/", IMG_STACK_200, RED),
    ("CALVIN / closed loop", "artifacts/server_media/\nserver:/data/senwang/data/calvin/rollouts/", IMG_SERVER_DRAWER_E4_245, ORANGE),
]
for i, (label_txt, path_txt, img, col) in enumerate(index_rows):
    yy = 1.62 + i * 0.96
    row_h = 0.78
    shape_rect(s, 0.72, yy, 8.05, row_h, PANEL_ALT if i % 2 else PANEL, 0.09, GRID, 0.7)
    shape_rect(s, 0.72, yy, 0.08, row_h, col)
    add_text(s, label_txt, 1.02, yy + 0.13, 2.05, 0.22, 12.2, col, True)
    add_text(s, path_txt, 3.15, yy + 0.10, 5.28, 0.43, 9.4, WHITE, False, "Consolas")
    add_image_cover(s, img, 9.12, yy + 0.05, 1.95, 0.68, frame=True)
    pill(s, ["rollout", "A/B trace", "paired replay", "closed loop", "CALVIN"][i], 11.22, yy + 0.25, 1.12, col, NAVY, 8.0)
shape_rect(s, 0.72, 6.52, 11.90, 0.28, PANEL, 0.08, GRID, 0.6)
add_text(s, "本 PPT 嵌入代表片段；服务器原始 MP4、episode JSON、metrics 和审计报告仍按上面的路径保存。", 0.98, 6.59, 11.35, 0.14, 10.8, WHITE, True, FONT_CN, PP_ALIGN.CENTER)
add_footer(s, "workspace artifacts / runbooks", "appendix")


OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
try:
    shutil.rmtree(_POSTER_DIR)
except Exception:
    pass
print(str(OUT))
