"""Thin entrypoint for direct token-by-token VAE reconstruction."""

from utils.training.vae.evaluation.runner import load_task_config, run


def main() -> None:
    cfg = load_task_config("configs/vae_eval_stream.yaml")
    run(cfg, mode="stream")


if __name__ == "__main__":
    main()
