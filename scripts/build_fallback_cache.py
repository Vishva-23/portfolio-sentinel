"""
Pre-populate fallback_cache.json for offline / demo resilience.

Run once before a demo or CI run:

    python scripts/build_fallback_cache.py

The script fetches 400 days of adjusted-close prices and the info dict for
each ticker in the default portfolio (plus the S&P 500 benchmark) and writes
them to fallback_cache.json in the project root.

The file is listed in .gitignore and should not be committed to version
control, as it contains live market data with a specific timestamp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the project root or from the scripts/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.market_data import get_prices, get_info  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "CRH"]
BENCHMARK_TICKER = "^GSPC"
LOOKBACK_DAYS = 400

ALL_TICKERS = DEFAULT_TICKERS + [BENCHMARK_TICKER]


def main() -> None:
    cache: dict = {}
    success: list[str] = []
    failures: list[str] = []

    for ticker in ALL_TICKERS:
        prices_key = f"prices_{ticker}_{LOOKBACK_DAYS}"
        info_key = f"info_{ticker}"

        print(f"  Fetching prices for {ticker} ({LOOKBACK_DAYS} days)...", end=" ")
        try:
            series = get_prices(ticker, LOOKBACK_DAYS)
            cache[prices_key] = {
                "dates": [d.strftime("%Y-%m-%d") for d in series.index],
                "values": [float(v) for v in series.values],
            }
            print(f"OK — {len(series)} rows")
            success.append(f"prices:{ticker}")
        except Exception as exc:
            print(f"FAILED — {exc}")
            failures.append(f"prices:{ticker}")

        print(f"  Fetching info for {ticker}...", end=" ")
        try:
            info = get_info(ticker)
            # Store only serialisable scalar values to keep the file compact
            serialisable = {
                k: v
                for k, v in info.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            cache[info_key] = serialisable
            print(f"OK — {len(serialisable)} fields")
            success.append(f"info:{ticker}")
        except Exception as exc:
            print(f"FAILED — {exc}")
            failures.append(f"info:{ticker}")

    output_path = PROJECT_ROOT / "fallback_cache.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"  fallback_cache.json written to: {output_path}")
    print(f"  Successful entries : {len(success)}")
    if failures:
        print(f"  Failed entries     : {len(failures)}")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  No failures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
