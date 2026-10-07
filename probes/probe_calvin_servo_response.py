"""Identify CALVIN one-observation servo response from independent air motions.

No language, object IDs, task success or checkpoint is used to fit the controller.
Calibration is an actuator experiment, not a task success benchmark or policy fix.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from clearvla.benchmarks.calvin_eval import _environment,_official_task_assets

def delta(now,before):
    d=np.asarray(now[:6],float)-np.asarray(before[:6],float)
    d[3:]=(d[3:]+np.pi)%(2*np.pi)-np.pi
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--state-json",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=False)
    official,*_= _official_task_assets()
    robot,scene=official.get_env_state_for_initial_condition(json.loads(a.state_json.read_text()))
    env=_environment(a.dataset_root,show_gui=False)
    positions=[np.array([0.,0.,0.]),np.array([.045,0.,.025]),np.array([-.045,.025,.015])]
    rows=[];tests=[];rng=np.random.default_rng(934701)
    scales=np.array([env.robot.max_rel_pos*env.robot.magic_scaling_factor_pos]*3+
                    [env.robot.max_rel_orn*env.robot.magic_scaling_factor_orn]*3)
    assert np.allclose(scales,[.02]*3+[.05]*3)
    def reset(offset):
        env.reset(robot_obs=robot,scene_obs=scene)
        env.robot.use_target_pose=True
        env.robot.target_pos=env.robot.target_pos+offset
        obs=env.get_obs()
        for _ in range(32):obs,_,_,_=env.step(np.r_[np.zeros(6),1.].copy())
        env.robot.use_target_pose=False
        return obs,np.zeros(6)
    def sequence():
        # Balanced, multi-frequency air motion; amplitudes independent of tasks.
        first=rng.uniform(-.12,.12,48)
        return np.r_[first,-first[::-1]]
    try:
        for pose in [0,1]:
            for axis in range(6):
                obs,previous=reset(positions[pose])
                for command in sequence():
                    native=np.zeros(6);native[axis]=command
                    nxt,_,_,info=env.step(np.r_[native,1.].copy())
                    actual=delta(nxt["robot_obs"],obs["robot_obs"])
                    legal=bool(not info["robot_info"]["contacts"] and min(obs["robot_obs"][2],nxt["robot_obs"][2])>.51)
                    rows.append({"pose":pose,"axis":axis,"previous":previous.tolist(),"request":(native*scales).tolist(),"actual":actual.tolist(),"legal":legal})
                    previous=actual;obs=nxt
                print(json.dumps({"phase":"calibration","pose":pose,"axis":axis}),flush=True)
        valid=[r for r in rows if r["legal"]]
        prev=np.asarray([r["previous"] for r in valid]);u=np.asarray([r["request"] for r in valid]);y=np.asarray([r["actual"] for r in valid])
        coefficients=[]
        for k in range(6):
            x=np.column_stack([prev[:,k],u[:,k],np.ones(len(y))])
            w=np.linalg.lstsq(x,y[:,k],rcond=None)[0]
            if not .05<float(w[1])<1.5:raise ValueError("unidentified actuator response")
            coefficients.append(w)
        coeff=np.asarray(coefficients)
        for mode in ["measured","inverse"]:
            for axis in range(6):
                obs,previous=reset(positions[2])
                # Same held-out desired sequence for both actuator modes.
                held=np.random.default_rng(101+axis).uniform(-.10,.10,48)
                held=np.r_[held,-held[::-1]]
                for value in held:
                    native=np.zeros(6);native[axis]=value
                    desired=native*scales
                    goal_delta=desired if mode=="measured" else (desired-coeff[:,0]*previous-coeff[:,2])/coeff[:,1]
                    # Admit no persistent controller target; this diagnostic
                    # inverse uses only the exact previous measured delta.
                    if np.max(np.abs(goal_delta/scales))>1:
                        raise ValueError("air-motion calibration exceeded declared native step bound")
                    pose=np.asarray(obs["robot_obs"][:6],float)+goal_delta
                    nxt,_,_,info=env.step([pose[:3],pose[3:],1])
                    actual=delta(nxt["robot_obs"],obs["robot_obs"])
                    tests.append({"mode":mode,"axis":axis,"desired":desired.tolist(),"actual":actual.tolist(),
                                  "goal_delta":goal_delta.tolist(),"contact":bool(info["robot_info"]["contacts"])})
                    previous=actual;obs=nxt
                print(json.dumps({"phase":"held-out","mode":mode,"axis":axis}),flush=True)
        summary={"scope":"independent task-free air-motion calibration; no closed-loop policy efficacy claim",
          "seed":934701,"model":"actual_delta = a*previous_measured_delta + b*goal_delta + c",
          "scales":scales.tolist(),"calibration_rows":len(rows),"legal_rows":len(valid),
          "coefficients":[{"previous_delta":float(w[0]),"goal_delta":float(w[1]),"bias":float(w[2])} for w in coeff],
          "held_out":{}}
        for mode in ["measured","inverse"]:
            r=[r for r in tests if r["mode"]==mode]
            desired=np.asarray([v["desired"] for v in r]);actual=np.asarray([v["actual"] for v in r])
            summary["held_out"][mode]={"rms_error_by_axis":np.sqrt(((actual-desired)**2).mean(0)).tolist(),
              "actual_to_requested_ls_gain":((actual*desired).sum(0)/(desired**2).sum(0)).tolist(),
              "contacts":sum(v["contact"] for v in r)}
        (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        (a.output/"samples.json").write_text(json.dumps({"calibration":rows,"held_out":tests})+"\n")
        (a.output/"complete.json").write_text(json.dumps({"complete":True,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+"\n")
        print(json.dumps(summary),flush=True)
    finally:env.close()
if __name__=="__main__":main()
