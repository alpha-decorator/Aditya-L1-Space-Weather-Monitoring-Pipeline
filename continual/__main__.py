"""python -m continual {seed,ingest,daily,status}"""

import argparse
import json

from . import ROOT, DataStore, Registry


def main():
    ap = argparse.ArgumentParser(prog="python -m continual")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="import the NCEI archive downloaded by nowcast.ipynb into the store")
    s.add_argument("--archive", default=str(ROOT / "nowcast" / "data" / "goes" / "avg1m"))
    sub.add_parser("ingest", help="store new complete days from the real-time feed")
    d = sub.add_parser("daily", help="ingest + retrain + promote if better")
    d.add_argument("--no-ingest", action="store_true")
    sub.add_parser("status", help="store coverage, active models, last decisions")
    a = ap.parse_args()

    store, reg = DataStore(), Registry()
    if a.cmd == "seed":
        print(f"filled {store.seed_from_archive(a.archive):,} minutes")
        print(json.dumps(store.status(), indent=2))
    elif a.cmd == "ingest":
        print(json.dumps(store.ingest_live(), indent=2))
    elif a.cmd == "daily":
        from .retrain import daily_cycle
        daily_cycle(store, reg, ingest=not a.no_ingest)
    elif a.cmd == "status":
        print(json.dumps(store.status(), indent=2))
        for t in ("C1", "M1"):
            print(f"{t}: active {reg.active_version(t)} (trained through {reg.train_end(t)})")
        for e in reg.history(6):
            print(f"  {e['run_at']} {e['target']}: {e['reason']}")


if __name__ == "__main__":
    main()
