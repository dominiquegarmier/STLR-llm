from __future__ import annotations

from stlr.model import STLR


def main() -> int:
    n_params = sum(p.numel() for p in STLR().parameters() if p.requires_grad)
    print(f'Number of parameters: {n_params:,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
