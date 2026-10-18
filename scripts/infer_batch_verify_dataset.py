import time

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "taruki/gemma4-infer-batch-v1-adapters"

api = KaggleApi()
api.authenticate()

for attempt in range(1, 9):
    try:
        files = api.dataset_list_files(DATASET).files
        print("dataset readable, %d files:" % len(files))
        for f in sorted(files, key=lambda x: x.name):
            print("  %-52s %d B" % (f.name, f.total_bytes))
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print("attempt %d: %s: %s" % (attempt, type(exc).__name__, exc), flush=True)
        if attempt < 8:
            time.sleep(20)

print("DATASET NOT READABLE")
raise SystemExit(1)
