import argparse

from .config import load_config
from .pipeline import run


def _print_table(title, df, score_col):
    print(f"\n=== {title} ===")

    if df.empty:
        print("今日無符合條件標的")
        return

    columns = [
        c
        for c in [
            "ticker",
            "name",
            score_col,
        ]
        if c in df.columns
    ]

    print(
        df[columns]
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["run"],
    )

    args = parser.parse_args()

    if args.command == "run":

        result = run(
            load_config()
        )

        _print_table(
            "STRONG",
            result["strong"],
            "score",
        )

        _print_table(
            "BREAKOUT",
            result["breakout"],
            "breakout_score",
        )

        _print_table(
            "IGNITION",
            result["ignition"],
            "ignition_score",
        )


if __name__ == "__main__":
    main()
