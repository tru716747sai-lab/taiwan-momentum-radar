import argparse, json
from .config import load_config
from .pipeline import run
from .backtest import run_backtest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run","backtest"])
    p.add_argument("--config", default="config/settings.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.command == "run":
        top = run(cfg)
        print(top[["ticker","name","score"]].to_string(index=False))
    else:
        print(json.dumps(run_backtest(cfg), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
