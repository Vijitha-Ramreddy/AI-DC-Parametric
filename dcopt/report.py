"""Aggregate methodology reports."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(f"Report module scaffolded with config {args.config}.")


if __name__ == "__main__":
    main()
