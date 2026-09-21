from pathlib import Path
p=Path('make_recent_work_ppt.py')
s=p.read_text(encoding='utf-8')
s=s.replace("OUT = os.path.join(ROOT, 'artifacts', 'clearvla_recent_work_summary_20260913.pptx')","OUT = os.path.join(ROOT, 'artifacts', 'clearvla_recent_work_summary_20260913_v2_scope_update.pptx')")
s=s.replace("IMG_B = os.path.join(ROOT,'artifacts','libero_causal_ab_20260910','B','short_probe_trace_v3_20260910','videos','contact_sheet.png')","IMG_B = os.path.join(ROOT,'artifacts','libero_causal_ab_20260910','B','short_probe_trace_v3_20260910','videos','contact_sheet.png')\nIMG_STACK = os.path.join(ROOT,'runs','stackcube_v2_eval_20260910','closed_loop_e2_seed1000001','step_0200.png')")
s=s.replace("stat(s,'主任务','LIBERO task 0',3.6,3.45,2.6,ORANGE,'black bowl → plate')","stat(s,'模拟环境','LIBERO + StackCube',3.6,3.45,2.6,ORANGE,'本周新搭建/打通')")
s=s.replace("textbox(s,'LIBERO 空间任务：因果 A/B、时序归因、专家交接与 release-first',0.72,2.08,10.4,0.45,18,MUTED,font=FONT_CN)","textbox(s,'两个新模拟环境 + 闭环诊断 + residual RL 准备',0.72,2.08,10.4,0.45,18,MUTED,font=FONT_CN)")
# Insert scope slide before exec slide
marker='# 2 executive readout\n'
insert='''# 2 scope\ns=prs.slides.add_slide(blank); base(s,'THIS WEEK SCOPE','本周工作范围：两个新模拟环境，外加一条 RL 训练准备线',2)\nrect(s,0.75,1.5,5.75,4.9,PANEL,True,line=GRID)\ntextbox(s,'01 · LIBERO',1.05,1.85,2.2,0.28,18,CYAN,True,font=FONT_CN)\ntextbox(s,'空间任务闭环',1.05,2.25,2.6,0.3,16,WHITE,True,font=FONT_CN)\nbullet(s,'remote bridge / Schema30 bring-up',1.1,2.85,4.65,size=15)\nbullet(s,'causal A/B、timing、expert handoff',1.1,3.35,4.65,size=15,accent=ORANGE)\nbullet(s,'release-first 训练与视频证据',1.1,3.85,4.65,size=15,accent=GREEN)\nlabel(s,'已形成机制证据',1.1,4.65,1.75,fill=CYAN,color=NAVY)\nrect(s,6.8,1.5,5.8,4.9,PANEL2,True,line=GRID)\ntextbox(s,'02 · StackCube + RL',7.1,1.85,3.5,0.28,18,ORANGE,True,font=FONT_CN)\ntextbox(s,'新环境打通，RL 进入工作范围',7.1,2.25,4.8,0.3,16,WHITE,True,font=FONT_CN)\nbullet(s,'ManiSkill StackCube-v1 / panda_wristcam',7.15,2.85,4.8,size=15)\nbullet(s,'数据、split、normalizer 与闭环 replay 可审计',7.15,3.35,4.8,size=15,accent=GREEN)\nbullet(s,'residual-SAC：config / replay / runner / checkpoint 接口已具备',7.15,3.85,4.8,size=15,accent=ORANGE)\nlabel(s,'RL 成绩尚未作为结论',7.15,4.65,2.2,fill=ORANGE,color=NAVY)\ntextbox(s,'网络结构修改只作为支撑背景，本次汇报不展开。',1.0,6.65,11.3,0.25,14,MUTED,True,align=PP_ALIGN.CENTER,font=FONT_CN)\n\n'''
s=s.replace(marker,insert+marker)
# Insert stackcube slide before causal slide
marker2='# 5 causal AB\n'
insert2='''# 5 stackcube + RL\ns=prs.slides.add_slide(blank); base(s,'STACKCUBE + RL · 09/09—09/10','StackCube 已从数据审计走到闭环 replay；RL 具备接入条件，但还没有宣称训练收益',5)\nadd_image(s,IMG_STACK,0.75,1.55,4.1,3.1)\nlabel(s,'StackCube-v1 · 400-step replay',1.35,4.82,2.7,fill=ORANGE,color=NAVY)\nrect(s,5.25,1.55,7.35,4.7,PANEL,True,line=GRID)\ntextbox(s,'已验证',5.6,1.9,1.2,0.25,15,CYAN,True,font=FONT_CN)\nstat(s,'数据','59 / 7 / 7',5.6,2.35,2.1,CYAN,'train / val / test episodes')\nstat(s,'归一化 RMSE','0.472',7.9,2.35,2.1,ORANGE,'StackCube E3 audit')\nstat(s,'闭环结果','0 reward',10.2,2.35,2.1,RED,'400 steps · seed1000001')\ntextbox(s,'诊断信号',5.6,3.65,1.3,0.25,15,ORANGE,True,font=FONT_CN)\nbullet(s,'min TCP→cube distance = 83.3 mm',5.65,4.08,6.3,size=15,accent=WHITE)\nbullet(s,'min cube→goal distance = 250.4 mm',5.65,4.55,6.3,size=15,accent=WHITE)\nbullet(s,'E6 专家前缀配对：第 44 步抓到红块，80 步内未完成叠放',5.65,5.02,6.3,size=15,accent=GREEN)\ntextbox(s,'RL 工作项：residual-SAC 配置、replay、runner、checkpoint 与 StackCube admission tests 已纳入范围；下一步才是 baseline gate → adapter train → 多 seed 评估。',5.6,5.65,6.45,0.4,13,MUTED,True,font=FONT_CN)\n\n'''
s=s.replace(marker2,insert2+marker2)
# shift page numbers for existing slides after inserted slides
for old,new in [(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)]:
    s=s.replace(f"base(s,'EXECUTIVE READOUT','这一轮最重要的变化，是把失败拆成可测的机制问题',{old})",f"base(s,'EXECUTIVE READOUT','这一轮最重要的变化，是把失败拆成可测的机制问题',{new})")
    # generic direct replacements for other known base lines
for title in ["WORKSTREAM MAP","CURRENT STACK","CAUSAL A/B · 09/10","TIMING ATTRIBUTION · 09/11","EXPERT HANDOFF · 09/11—09/13","RELEASE-FIRST · 09/13","WHAT WE LEARNED","EVIDENCE INDEX"]:
    pass
repls={
"'WORKSTREAM MAP','五个实验节点，把“跑不起来”拆成五个问题',3":"'WORKSTREAM MAP','五个实验节点，把“跑不起来”拆成五个问题',4",
"'CURRENT STACK','当前工作版本：v120 组件在 schema30 runtime 上收敛成可审计路径',4":"'CURRENT STACK','当前工作版本：v120 组件在 schema30 runtime 上收敛成可审计路径',5",
"'CAUSAL A/B · 09/10','修复坐标后，A/B 只显示“向目标靠近”，没有形成抓取闭环',5":"'CAUSAL A/B · 09/10','修复坐标后，A/B 只显示“向目标靠近”，没有形成抓取闭环',6",
"'TIMING ATTRIBUTION · 09/11','把夹爪关闭延迟 12 步，手臂仍走同一条轨迹',6":"'TIMING ATTRIBUTION · 09/11','把夹爪关闭延迟 12 步，手臂仍走同一条轨迹',7",
"'EXPERT HANDOFF · 09/11—09/13','36 步专家前缀，把同一个策略从“冷启动失败”带到“200 步成功”',7":"'EXPERT HANDOFF · 09/11—09/13','36 步专家前缀，把同一个策略从“冷启动失败”带到“200 步成功”',8",
"'RELEASE-FIRST · 09/13','r4 release-first 已跑通训练与审计，但仍处在“早期可用、未稳态”阶段',8":"'RELEASE-FIRST · 09/13','r4 release-first 已跑通训练与审计，但仍处在“早期可用、未稳态”阶段',9",
"'WHAT WE LEARNED','下一轮实验应该把预算放在“早期状态进入”和“长程闭环”',9":"'WHAT WE LEARNED','下一轮实验应该把预算放在“早期状态进入”和“长程闭环”',10",
"'EVIDENCE INDEX','素材与证据索引：PPT 只保留结论，原始视频/日志仍在工作区',10":"'EVIDENCE INDEX','素材与证据索引：PPT 只保留结论，原始视频/日志仍在工作区',11",
}
for a,b in repls.items(): s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
