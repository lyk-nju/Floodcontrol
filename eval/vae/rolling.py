"""Run finite-history replay rolling VAE reconstruction."""

from .evaluate_reconstruction import load_task_config, run


def main() -> None:
    cfg = load_task_config("eval/vae/rolling.yaml")
    run(cfg, mode="rolling")


if __name__ == "__main__":
    main()
