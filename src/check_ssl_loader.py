import argparse

from config import load_configs
from data import build_ssl_loaders
from seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["fixmatch", "mixmatch"], required=True)
    args = parser.parse_args()

    cfg = load_configs(args.config)
    set_seed(cfg["seed"])

    labeled_loader, unlabeled_loader, _, _ = build_ssl_loaders(cfg, args.method)
    x_l, y_l = next(iter(labeled_loader))
    u1, u2 = next(iter(unlabeled_loader))

    print("labeled x:", x_l.shape)
    print("labeled y:", y_l.shape)
    print("unlabeled 1:", u1.shape)
    print("unlabeled 2:", u2.shape)

if __name__ == "__main__":
    main()