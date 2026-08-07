"""
===========================================================
Project Title : SemCIDS
Author       : Madan Dahal
Date Created : June 15, 2026
===========================================================

run_all.py — Master pipeline for SemCIDS
Usage:
    python run_all.py                  # run everything
    python run_all.py --from 05        # restart from training
    python run_all.py --only 07        # run only main plots
    python run_all.py --skip 02,03     # skip steps
"""

import subprocess, sys, os, time, argparse, shutil

PIPELINE = [
    ("01","01_data_loader.py",
     "Load and preprocess CICIoT2023"),
    ("02","02_semantic_encoder.py",
     "Train task-oriented encoder + 4-method comparison"),
    ("03","03_environment.py",
     "Build joint states and  model"),
    ("04","04_drl_agent.py",
     "Define Dueling DDQN and Zhao networks"),
    ("05","05_train.py",
     "Train Algorithms"),
    ("06","06_evaluate.py",
     "Evaluate all 9 methods on test set"),
    ("07","07_plots_main.py",
     "Main comparison bar charts with error bars"),
    ("08","08_plots_extended.py"),
]

def setup():
    """Create required folders and copy drl_agent.py."""
    for d in ["Images","checkpoints","results"]:
        os.makedirs(d,exist_ok=True)
    # Python cannot import files starting with digits
    # Copy 04_drl_agent.py → drl_agent.py
    if os.path.exists("04_drl_agent.py"):
        shutil.copy("04_drl_agent.py","drl_agent.py")
        print("  [setup] Copied 04_drl_agent.py → drl_agent.py")

def run_module(script, description):
    print(f"\n{'═'*65}")
    print(f"  Running : {script}")
    print(f"  Purpose : {description}")
    print(f"{'═'*65}")
    t0=time.time()
    subprocess.run([sys.executable,script],check=True)
    elapsed=time.time()-t0
    print(f"\n  ✓  {script} completed in {elapsed:.1f}s")
    return elapsed

def main():
    parser=argparse.ArgumentParser(
        description="SemCIDS full pipeline")
    parser.add_argument("--from",dest="from_step",default=None,
                        help="Start from step e.g. --from 05")
    parser.add_argument("--only",dest="only_step",default=None,
                        help="Run only one step e.g. --only 09")
    parser.add_argument("--skip",dest="skip_steps",default="",
                        help="Skip steps e.g. --skip 02,03")
    args=parser.parse_args()
    skip=set(args.skip_steps.split(",")) if args.skip_steps else set()

    setup()

    print("\n"+"█"*65)
    print("  SemCIDS — Multi-Device IDS Semantic Offloading")
    print("  Steps: 01-08 core pipeline")
    print("         09    FNR/FPR analysis by threat priority")
    print("         10    3D reward surface + weight sensitivity")
    print("█"*65)

    total=0; ran=0
    for sid,script,desc in PIPELINE:
        if args.only_step and sid!=args.only_step: continue
        if args.from_step and sid<args.from_step:
            print(f"\n  [SKIP] {script}"); continue
        if sid in skip:
            print(f"\n  [SKIP] {script}"); continue
        if not os.path.exists(script):
            print(f"\n  [WARN] {script} not found — skipping")
            continue
        total+=run_module(script,desc)
        ran+=1

    print(f"\n{'█'*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Steps run    : {ran}")
    print(f"  Total time   : {total:.1f}s")
    print(f"  Plots        → Images/")
    print(f"  Models       → checkpoints/")
    print(f"  Results/CSV  → results/")
    print(f"{'█'*65}\n")

if __name__=="__main__":
    main()
