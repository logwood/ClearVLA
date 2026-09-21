from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.enum.text import MSO_VERTICAL_ANCHOR
import os, math

ROOT = r'C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts'
OUT = os.path.join(ROOT, 'artifacts', 'clearvla_recent_work_summary_20260913_v2_scope_update.pptx')
IMG_EXPERT = os.path.join(ROOT,'artifacts','libero_release_first_20260913','r4_releasefirst','expert_handoff_v4_200s_prefix36_retry2_expert_contact_sheet.jpg')
IMG_COLD = os.path.join(ROOT,'artifacts','libero_release_first_20260913','r4_releasefirst','expert_handoff_v4_200s_prefix36_retry2_cold_contact_sheet.jpg')
IMG_A = os.path.join(ROOT,'artifacts','libero_causal_ab_20260910','A','short_probe_trace_v3_20260910','videos','contact_sheet.png')
IMG_B = os.path.join(ROOT,'artifacts','libero_causal_ab_20260910','B','short_probe_trace_v3_20260910','videos','contact_sheet.png')
IMG_STACK = os.path.join(ROOT,'runs','stackcube_v2_eval_20260910','closed_loop_e2_seed1000001','step_0200.png')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

NAVY = RGBColor(9, 18, 34)
PANEL = RGBColor(17, 31, 53)
PANEL2 = RGBColor(22, 42, 69)
WHITE = RGBColor(242, 247, 252)
MUTED = RGBColor(170, 189, 211)
CYAN = RGBColor(67, 208, 224)
ORANGE = RGBColor(255, 170, 74)
RED = RGBColor(245, 104, 104)
GREEN = RGBColor(88, 214, 151)
GRID = RGBColor(49, 72, 100)
FONT = 'Aptos'
FONT_CN = 'Microsoft YaHei'


def rect(slide,x,y,w,h,fill, radius=False, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line if line else fill
    if radius:
        shape.adjustments[0] = 0.08
    return shape

def line(slide,x1,y1,x2,y2,color=GRID,width=1.2,dash=None):
    s=slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    s.line.color.rgb=color; s.line.width=Pt(width)
    if dash: s.line.dash_style=dash
    return s

def textbox(slide,text,x,y,w,h,size=18,color=WHITE,bold=False,align=PP_ALIGN.LEFT,font=FONT,margin=0.05,valign=MSO_ANCHOR.TOP):
    tb=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=Inches(margin); tf.margin_right=Inches(margin); tf.margin_top=Inches(margin); tf.margin_bottom=Inches(margin)
    tf.vertical_anchor=valign
    p=tf.paragraphs[0]; p.alignment=align
    run=p.add_run(); run.text=text
    run.font.name=font; run.font.size=Pt(size); run.font.bold=bold; run.font.color.rgb=color
    return tb

def richbox(slide,runs,x,y,w,h,align=PP_ALIGN.LEFT,margin=0.05,valign=MSO_ANCHOR.TOP):
    tb=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=Inches(margin); tf.margin_right=Inches(margin); tf.margin_top=Inches(margin); tf.margin_bottom=Inches(margin); tf.vertical_anchor=valign
    p=tf.paragraphs[0]; p.alignment=align
    for txt,size,col,bold in runs:
        r=p.add_run(); r.text=txt; r.font.name=FONT; r.font.size=Pt(size); r.font.color.rgb=col; r.font.bold=bold
    return tb

def base(slide, kicker, title, page):
    bg=slide.background.fill; bg.solid(); bg.fore_color.rgb=NAVY
    rect(slide,0,0,13.333,0.08,CYAN)
    textbox(slide,kicker.upper(),0.55,0.28,4.4,0.25,10,CYAN,True)
    textbox(slide,title,0.55,0.62,12.1,0.55,26,WHITE,True,font=FONT_CN)
    textbox(slide,f'{page:02d}',12.35,0.35,0.45,0.25,10,MUTED,True,align=PP_ALIGN.RIGHT)

def bullet(slide, text, x,y,w, color=WHITE, size=16, accent=CYAN):
    rect(slide,x,y+0.12,0.09,0.09,accent,True)
    textbox(slide,text,x+0.18,y,w-0.18,0.45,size,color,font=FONT_CN)

def stat(slide,label,value,x,y,w,accent=CYAN,sub=''):
    rect(slide,x,y,w,0.92,PANEL,True,line=GRID)
    textbox(slide,label.upper(),x+0.16,y+0.12,w-0.32,0.2,9,MUTED,True)
    textbox(slide,value,x+0.16,y+0.34,w-0.32,0.34,24,accent,True)
    if sub: textbox(slide,sub,x+0.16,y+0.72,w-0.32,0.15,9,MUTED)

def add_image(slide,path,x,y,w,h):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    rect(slide,x,y,w,h,PANEL,True,line=GRID); textbox(slide,'image unavailable',x,y+h/2-0.12,w,0.2,10,MUTED,align=PP_ALIGN.CENTER)

def label(slide,txt,x,y,w,fill=PANEL2,color=WHITE):
    rect(slide,x,y,w,0.28,fill,True)
    textbox(slide,txt,x+0.08,y+0.03,w-0.16,0.18,10,color,True,align=PP_ALIGN.CENTER,font=FONT_CN)

# 1 cover
s=prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
rect(s,0,0,13.333,0.12,CYAN)
textbox(s,'CLEARVLA / RECENT WORK',0.7,0.62,5,0.3,11,CYAN,True)
textbox(s,'从“能跑”到“证据闭环”',0.7,1.15,9.8,0.8,34,WHITE,True,font=FONT_CN)
textbox(s,'两个新模拟环境 + 闭环诊断 + residual RL 准备',0.72,2.08,10.4,0.45,18,MUTED,font=FONT_CN)
line(s,0.75,3.05,12.3,3.05,GRID,1)
stat(s,'时间窗','09/08—09/13',0.75,3.45,2.6,CYAN,'最近一轮工作')
stat(s,'模拟环境','LIBERO + StackCube',3.6,3.45,2.6,ORANGE,'本周新搭建/打通')
stat(s,'关键视频','20+ MP4',6.45,3.45,2.6,GREEN,'contact sheet 已归档')
stat(s,'当前结论','短轨迹可归因',9.3,3.45,2.6,CYAN,'正式成功率仍待长程验证')
textbox(s,'工作区扫描版 · 证据优先 · 2026-09-13',0.75,6.75,6,0.25,10,MUTED,font=FONT_CN)

# 2 scope
s=prs.slides.add_slide(blank); base(s,'THIS WEEK SCOPE','本周工作范围：两个新模拟环境，外加一条 RL 训练准备线',2)
rect(s,0.75,1.5,5.75,4.9,PANEL,True,line=GRID)
textbox(s,'01 · LIBERO',1.05,1.85,2.2,0.28,18,CYAN,True,font=FONT_CN)
textbox(s,'空间任务闭环',1.05,2.25,2.6,0.3,16,WHITE,True,font=FONT_CN)
bullet(s,'remote bridge / Schema30 bring-up',1.1,2.85,4.65,size=15)
bullet(s,'causal A/B、timing、expert handoff',1.1,3.35,4.65,size=15,accent=ORANGE)
bullet(s,'release-first 训练与视频证据',1.1,3.85,4.65,size=15,accent=GREEN)
label(s,'已形成机制证据',1.1,4.65,1.75,fill=CYAN,color=NAVY)
rect(s,6.8,1.5,5.8,4.9,PANEL2,True,line=GRID)
textbox(s,'02 · StackCube + RL',7.1,1.85,3.5,0.28,18,ORANGE,True,font=FONT_CN)
textbox(s,'新环境打通，RL 进入工作范围',7.1,2.25,4.8,0.3,16,WHITE,True,font=FONT_CN)
bullet(s,'ManiSkill StackCube-v1 / panda_wristcam',7.15,2.85,4.8,size=15)
bullet(s,'数据、split、normalizer 与闭环 replay 可审计',7.15,3.35,4.8,size=15,accent=GREEN)
bullet(s,'residual-SAC：config / replay / runner / checkpoint 接口已具备',7.15,3.85,4.8,size=15,accent=ORANGE)
label(s,'RL 成绩尚未作为结论',7.15,4.65,2.2,fill=ORANGE,color=NAVY)
textbox(s,'网络结构修改只作为支撑背景，本次汇报不展开。',1.0,6.65,11.3,0.25,14,MUTED,True,align=PP_ALIGN.CENTER,font=FONT_CN)

# 2 executive readout
s=prs.slides.add_slide(blank); base(s,'EXECUTIVE READOUT','这一轮最重要的变化，是把失败拆成可测的机制问题',11)
rect(s,0.55,1.45,7.9,4.95,PANEL,True,line=GRID)
textbox(s,'三条已被证据支持的判断',0.85,1.75,5.4,0.3,16,CYAN,True,font=FONT_CN)
bullet(s,'坐标索引修复后，A/B 短闭环均为 0/4 成功；此前约 2 cm “推碗”现象是状态索引伪象。',0.9,2.25,6.95,size=17,accent=RED)
bullet(s,'延迟关闭夹爪 12 步，手臂动作与 EEF 轨迹几乎不变：夹爪时序不是手臂控制的主因。',0.9,3.15,6.95,size=17,accent=ORANGE)
bullet(s,'36 步专家前缀 + 164 步策略在 200 步 rollout 成功；cold-start 同配置失败，说明“状态进入/早期轨迹”是当前瓶颈。',0.9,4.05,6.95,size=17,accent=GREEN)
textbox(s,'这不是官方 LIBERO 分数，而是一组能指导下一轮实验的机制证据。',0.9,5.45,6.8,0.35,15,MUTED,True,font=FONT_CN)
rect(s,8.75,1.45,3.95,4.95,PANEL2,True,line=GRID)
textbox(s,'最新 release-first 快照',9.05,1.75,3.2,0.3,15,CYAN,True,font=FONT_CN)
stat(s,'训练','epoch 1 / step 120',9.05,2.25,3.35,WHITE,'schema30 · batch 8')
stat(s,'事件 F1','0.119',9.05,3.35,3.35,ORANGE,'precision 0.063 · recall 1.0')
stat(s,'proposal RMSE','0.201 m',9.05,4.45,3.35,GREEN,'validation physical')

# 3 timeline
s=prs.slides.add_slide(blank); base(s,'WORKSTREAM MAP','五个实验节点，把“跑不起来”拆成五个问题',4)
# timeline
x0=0.95; y=2.15; xs=[1.0,3.35,5.7,8.05,10.4]
line(s,1.0,y,11.75,y,GRID,2)
items=[('09/08','Bring-up','Schema30 + remote bridge\n语言 bank / DINO ABI','基础设施',CYAN),('09/10','Causal A/B','修复 qpos flatten\n±3 cm bowl shift','坐标与反馈',ORANGE),('09/10','Retarget','release-centered\ncounterfactual','夹爪边界',RED),('09/11','Timing + handoff','delay-close / 36-step\nexpert prefix','时序归因',GREEN),('09/13','Release-first','r4 training + 200-step\nretry2 evidence','下一版基线',CYAN)]
for i,(d,t,desc,tag,c) in enumerate(items):
    cx=xs[i]; rect(s,cx-0.11,y-0.11,0.22,0.22,c,True)
    textbox(s,d,cx-0.55,y-0.6,1.1,0.25,10,c,True,align=PP_ALIGN.CENTER)
    rect(s,cx-0.92,2.55,1.84,2.15,PANEL,True,line=GRID)
    textbox(s,t,cx-0.78,2.78,1.56,0.3,14,WHITE,True,align=PP_ALIGN.CENTER,font=FONT_CN)
    textbox(s,desc,cx-0.78,3.2,1.56,0.62,11,MUTED,align=PP_ALIGN.CENTER,font=FONT_CN)
    label(s,tag,cx-0.65,4.05,1.3,fill=c,color=NAVY)
textbox(s,'证据链：输入与状态 → 反馈灵敏度 → 夹爪时序 → 早期状态 → release-first 训练',1.0,5.55,11.2,0.35,16,WHITE,True,align=PP_ALIGN.CENTER,font=FONT_CN)

# 4 iteration stack
s=prs.slides.add_slide(blank); base(s,'CURRENT STACK','当前工作版本：v120 组件在 schema30 runtime 上收敛成可审计路径',5)
# layered diagram
layers=[('Observation','三帧 top+wrist · Flow-DINO progressive G1/G2/G3','DINOv2-base · 768 dim · 336×336',CYAN),('Top / evidence','object-intent dynamics · dense grounding · typed P1/P2 routes','4 object slots · 4 horizon intervals',ORANGE),('Bottom / execution','shared seed · terminal-layer contracts · continuous gripper field','MMDiT evidence blocks · capacity gate',GREEN),('Runtime / evaluation','24-row prediction · execute row 0 · physical chart + causal probes','state/action normalizers fingerprinted',WHITE)]
for i,(a,b,c,col) in enumerate(layers):
    yy=1.55+i*1.18; rect(s,0.8,yy,11.7,0.82,PANEL if i%2==0 else PANEL2,True,line=GRID)
    rect(s,1.0,yy+0.14,1.9,0.52,col,True)
    textbox(s,a,1.1,yy+0.27,1.7,0.22,14,NAVY,True,align=PP_ALIGN.CENTER,font=FONT_CN)
    textbox(s,b,3.2,yy+0.16,7.3,0.25,16,WHITE,True,font=FONT_CN)
    textbox(s,c,3.2,yy+0.48,7.3,0.18,10,MUTED,font=FONT_CN)
    textbox(s,f'{i+1}',11.7,yy+0.26,0.35,0.22,13,col,True,align=PP_ALIGN.CENTER)
textbox(s,'审计重点：每个模块都有 identity / normalizer / runtime 证据，避免“实验名 ≠ 实际图”。',0.95,6.5,11.4,0.3,14,MUTED,True,font=FONT_CN)

# 5 stackcube + RL
s=prs.slides.add_slide(blank); base(s,'STACKCUBE + RL · 09/09—09/10','StackCube 已从数据审计走到闭环 replay；RL 具备接入条件，但还没有宣称训练收益',5)
add_image(s,IMG_STACK,0.75,1.55,4.1,3.1)
label(s,'StackCube-v1 · 400-step replay',1.35,4.82,2.7,fill=ORANGE,color=NAVY)
rect(s,5.25,1.55,7.35,4.7,PANEL,True,line=GRID)
textbox(s,'已验证',5.6,1.9,1.2,0.25,15,CYAN,True,font=FONT_CN)
stat(s,'数据','59 / 7 / 7',5.6,2.35,2.1,CYAN,'train / val / test episodes')
stat(s,'归一化 RMSE','0.472',7.9,2.35,2.1,ORANGE,'StackCube E3 audit')
stat(s,'闭环结果','0 reward',10.2,2.35,2.1,RED,'400 steps · seed1000001')
textbox(s,'诊断信号',5.6,3.65,1.3,0.25,15,ORANGE,True,font=FONT_CN)
bullet(s,'min TCP→cube distance = 83.3 mm',5.65,4.08,6.3,size=15,accent=WHITE)
bullet(s,'min cube→goal distance = 250.4 mm',5.65,4.55,6.3,size=15,accent=WHITE)
bullet(s,'E6 专家前缀配对：第 44 步抓到红块，80 步内未完成叠放',5.65,5.02,6.3,size=15,accent=GREEN)
textbox(s,'RL 工作项：residual-SAC 配置、replay、runner、checkpoint 与 StackCube admission tests 已纳入范围；下一步才是 baseline gate → adapter train → 多 seed 评估。',5.6,5.65,6.45,0.4,13,MUTED,True,font=FONT_CN)

# 5 causal AB
s=prs.slides.add_slide(blank); base(s,'CAUSAL A/B · 09/10','修复坐标后，A/B 只显示“向目标靠近”，没有形成抓取闭环',6)
add_image(s,IMG_A,0.65,1.45,6.0,0.92); add_image(s,IMG_B,6.85,1.45,5.8,0.92)
label(s,'A · causal-prefix',1.0,2.48,1.65,fill=CYAN,color=NAVY); label(s,'B · terminal-suffix',7.15,2.48,1.85,fill=ORANGE,color=NAVY)
# metric cards
stat(s,'成功','0 / 4',0.8,2.95,2.0,RED,'两 checkpoint')
stat(s,'EEF 位移','0.066 m',2.95,2.95,2.0,CYAN,'A mean')
stat(s,'最小 EEF→碗','0.332 m',5.1,2.95,2.0,WHITE,'A mean')
stat(s,'XY 物体位移','5e−9 m',7.25,2.95,2.0,GREEN,'stationary')
stat(s,'目标 y 响应','1.01 mm',9.4,2.95,2.3,ORANGE,'60 mm shift → 1.69%')
rect(s,0.8,4.2,11.75,1.75,PANEL,True,line=GRID)
textbox(s,'解读',1.05,4.48,0.8,0.25,13,CYAN,True,font=FONT_CN)
textbox(s,'两组 rollout 都把 EEF 向碗推进了约 3.5–4.9 cm，但 8 步后仍离碗约 31–36 cm；碗的 XY 位移约 5×10⁻⁹ m，不能算接触或推动。',1.95,4.42,9.9,0.42,16,WHITE,font=FONT_CN)
textbox(s,'因此：反馈符号是对的，反馈增益太弱；“短 trace 的正确方向”与“可完成任务”之间仍有明显鸿沟。',1.95,5.02,9.9,0.35,14,MUTED,True,font=FONT_CN)

# 6 timing attribution
s=prs.slides.add_slide(blank); base(s,'TIMING ATTRIBUTION · 09/11','把夹爪关闭延迟 12 步，手臂仍走同一条轨迹',7)
rect(s,0.75,1.45,5.1,4.8,PANEL,True,line=GRID)
textbox(s,'实验设计',1.05,1.78,1.5,0.28,15,CYAN,True,font=FONT_CN)
bullet(s,'baseline：正常 bridge inference',1.1,2.3,4.2,size=15)
bullet(s,'delayed-close：前 12 步把正向夹爪命令置零',1.1,2.85,4.2,size=15,accent=ORANGE)
bullet(s,'arm[0:6] 每一步逐元素复制',1.1,3.4,4.2,size=15,accent=GREEN)
bullet(s,'32 步 deployment-only probe',1.1,3.95,4.2,size=15,accent=WHITE)
textbox(s,'两条件都没有物体接触，也都没有成功。',1.1,5.25,4.25,0.35,15,MUTED,True,font=FONT_CN)
# right metrics
textbox(s,'差异只出现在 gripper channel',6.25,1.78,5.2,0.28,15,ORANGE,True,font=FONT_CN)
stat(s,'Arm action Δ RMS','0.000',6.3,2.35,2.7,GREEN,'exact copy')
stat(s,'EEF trajectory Δ RMS','9.1 μm',9.2,2.35,2.7,CYAN,'post-step')
stat(s,'Gripper action Δ RMS','0.441',6.3,3.55,2.7,ORANGE,'12 close cmds suppressed')
stat(s,'Object Δ RMS','0.000',9.2,3.55,2.7,WHITE,'stationary')
rect(s,6.3,4.95,5.6,1.15,PANEL2,True,line=GRID)
textbox(s,'结论：夹爪命令时序会改变“命令流”，但没有改变手臂控制轨迹；优先排查早期状态、目标 grounding 与长程闭环。',6.55,5.22,5.1,0.55,15,WHITE,True,font=FONT_CN)

# 7 expert handoff
s=prs.slides.add_slide(blank); base(s,'EXPERT HANDOFF · 09/11—09/13','36 步专家前缀，把同一个策略从“冷启动失败”带到“200 步成功”',8)
add_image(s,IMG_COLD,0.7,1.45,5.95,1.55); add_image(s,IMG_EXPERT,6.8,1.45,5.85,1.55)
label(s,'cold-start · 0 / 200',2.25,3.12,1.9,fill=RED,color=NAVY); label(s,'expert prefix 36 + policy 164 · success',8.25,3.12,3.25,fill=GREEN,color=NAVY)
# bars distance
textbox(s,'最终 EEF→目标距离（xyz）',0.85,3.7,3.4,0.25,14,MUTED,True,font=FONT_CN)
for labeltxt,val,col,yy in [('cold',0.1047,RED,4.15),('handoff',0.1420,GREEN,4.7)]:
    textbox(s,labeltxt,0.9,yy-0.02,0.7,0.2,12,WHITE,True,font=FONT_CN)
    rect(s,1.7,yy,4.0,0.28,GRID,True)
    rect(s,1.7,yy,4.0*(1-val/0.18),0.28,col,True)
    textbox(s,f'{val:.3f} m',5.85,yy-0.04,0.8,0.25,12,col,True,align=PP_ALIGN.RIGHT)
rect(s,6.8,3.75,5.85,2.0,PANEL,True,line=GRID)
textbox(s,'关键对比',7.1,4.02,1.3,0.25,14,CYAN,True,font=FONT_CN)
bullet(s,'expert prefix：success = true',7.1,4.45,4.9,size=15,accent=GREEN)
bullet(s,'cold start：success = false',7.1,4.92,4.9,size=15,accent=RED)
bullet(s,'handoff 后策略动作仍被执行 164 步',7.1,5.39,4.9,size=15,accent=ORANGE)
textbox(s,'解释：专家前缀提供了可用的早期状态轨迹，但尚不能说明策略已具备稳定 cold-start 能力。',0.85,6.25,11.6,0.3,14,MUTED,True,font=FONT_CN)

# 8 release first
s=prs.slides.add_slide(blank); base(s,'RELEASE-FIRST · 09/13','r4 release-first 已跑通训练与审计，但仍处在“早期可用、未稳态”阶段',9)
# left mini chart losses
rect(s,0.75,1.45,6.0,4.95,PANEL,True,line=GRID)
textbox(s,'epoch 1 / step 120 · validation snapshot',1.05,1.75,5.2,0.25,14,CYAN,True,font=FONT_CN)
metrics=[('proposal RMSE',0.201,GREEN),('tail RMSE',0.198,CYAN),('gripper F1',0.119,ORANGE),('event precision',0.063,RED)]
maxv=0.22
for i,(lab,v,col) in enumerate(metrics):
    yy=2.35+i*0.72; textbox(s,lab,1.05,yy-0.02,1.55,0.2,11,WHITE,font=FONT_CN)
    rect(s,2.7,yy,2.9,0.22,GRID,True); rect(s,2.7,yy,2.9*v/maxv,0.22,col,True)
    textbox(s,f'{v:.3f}',5.78,yy-0.03,0.55,0.22,11,col,True,align=PP_ALIGN.RIGHT)
textbox(s,'事件 recall = 1.0，但 precision = 0.063；当前 gripper event 预测偏“过报”。',1.05,5.45,5.15,0.5,14,MUTED,True,font=FONT_CN)
# right cards
rect(s,7.05,1.45,5.6,4.95,PANEL2,True,line=GRID)
textbox(s,'训练健康度与边界',7.35,1.75,4.7,0.25,14,ORANGE,True,font=FONT_CN)
stat(s,'flow JEPA variance floor','2.3e−6',7.35,2.25,2.4,CYAN,'aligned G2 input')
stat(s,'gradient preclip max','5.59',9.95,2.25,2.4,ORANGE,'spike audit threshold 5.0')
stat(s,'proposal zero gain','0.000',7.35,3.35,2.4,GREEN,'neutral ablation')
stat(s,'motion F1','1.000',9.95,3.35,2.4,GREEN,'head-level event')
textbox(s,'读法：结构性探针多数通过，真正的任务闭环仍受早期轨迹与夹爪事件质量限制。',7.35,4.65,4.7,0.55,15,WHITE,True,font=FONT_CN)

# 9 learnings
s=prs.slides.add_slide(blank); base(s,'WHAT WE LEARNED','下一轮实验应该把预算放在“早期状态进入”和“长程闭环”',10)
cols=[(0.8,'已经解决',GREEN,['flattened-state 坐标契约可验证','v1/v2 错误结果已降级为 provenance','A/B 与 timing probe 的 identity 一致']),(4.55,'仍未解决',ORANGE,['cold-start 200 步失败','8 步短 trace 仍离目标很远','gripper event 过报，precision 低']),(8.3,'下一步',CYAN,['延长正式 rollout，分离 approach / grasp / place','把 handoff 变成可控初始化变量','围绕 early-state grounding 做 ablation'])]
for x,t,c,items in cols:
    rect(s,x,1.55,3.55,4.7,PANEL,True,line=GRID)
    rect(s,x,1.55,3.55,0.16,c)
    textbox(s,t,x+0.25,1.95,3.05,0.3,18,c,True,font=FONT_CN)
    for i,it in enumerate(items): bullet(s,it,x+0.28,2.65+i*0.85,2.95,size=14,accent=c)
textbox(s,'实验决策原则：先保留可证伪的诊断，再把成功视频转成可复现的初始化与闭环协议。',1.0,6.55,11.3,0.3,15,WHITE,True,align=PP_ALIGN.CENTER,font=FONT_CN)

# 10 appendix
s=prs.slides.add_slide(blank); base(s,'EVIDENCE INDEX','素材与证据索引：PPT 只保留结论，原始视频/日志仍在工作区',11)
rect(s,0.7,1.45,12.0,4.95,PANEL,True,line=GRID)
textbox(s,'主证据',1.0,1.75,1.4,0.25,14,CYAN,True,font=FONT_CN)
entries=[
('Causal A/B','artifacts/libero_causal_ab_20260910/README.md + v3_audit.json','4 MP4 / A+B contact sheet'),
('Timing attribution','artifacts/libero_timing_20260911/timing_probe_v1.json','baseline vs delayed-close'),
('Expert handoff','artifacts/libero_release_first_20260913/r4_releasefirst/expert_handoff_v4_200s_prefix36_retry2.json','expert + cold contact sheet'),
('Release-first','artifacts/libero_release_first_20260913/r4_releasefirst/metrics.jsonl','train / validation / numerics'),
('Retarget / release','artifacts/libero_retarget_20260910/release_centered_probe_v*.json','counterfactual gripper boundary'),
]
for i,(a,b,c) in enumerate(entries):
    yy=2.2+i*0.73; rect(s,1.0,yy,1.55,0.36,PANEL2,True); textbox(s,a,1.08,yy+0.08,1.4,0.18,11,WHITE,True,font=FONT_CN)
    textbox(s,b,2.8,yy+0.02,6.6,0.22,10,MUTED,font='Consolas')
    textbox(s,c,9.55,yy+0.02,2.55,0.22,10,CYAN,font=FONT_CN)
textbox(s,'视频视觉证据：expert_prefix / cold_start contact sheet、A/B v3 contact sheet，以及 timing probe 的 baseline / delayed_close MP4。',1.0,6.0,11.5,0.35,13,WHITE,True,font=FONT_CN)
textbox(s,'ClearVLA · recent work summary · 2026-09-13',1.0,6.7,5.5,0.2,9,MUTED,font=FONT_CN)

# save
os.makedirs(os.path.dirname(OUT),exist_ok=True)
prs.save(OUT)
print(OUT)
