"""Thin entrypoint for finite-history rolling VAE reconstruction."""

from utils.training.vae.evaluation.runner import load_task_config, run


def main() -> None:
    cfg = load_task_config("configs/vae_eval_rolling.yaml")
    run(cfg, mode="rolling")


if __name__ == "__main__":
    main()
