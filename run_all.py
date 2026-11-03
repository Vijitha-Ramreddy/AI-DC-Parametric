"""Run available project foundation CLIs."""

import subprocess


if __name__ == "__main__":
    for module in ("dcopt.io", "dcopt.phase0_data_foundation", "dcopt.baseline", "dcopt.settling",
                   "dcopt.ordering", "dcopt.predictive", "dcopt.adaptive", "dcopt.pat"):
        subprocess.run(["python", "-m", module, "--config", "config.yaml"], check=True)
