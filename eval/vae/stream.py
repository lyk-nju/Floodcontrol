"""Run direct token-by-token VAE stream reconstruction."""

from .evaluate_reconstruction import load_task_config, run


def main() -> None:
    cfg = load_task_config("eval/vae/stream.yaml")
    run(cfg, mode="stream")


if __name__ == "__main__":
    main()

