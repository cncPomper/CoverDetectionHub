import json
import pandas as pd

with open('./da-tacos_metadata/da-tacos_benchmark_subset_metadata.json') as f:
    data = json.load(f)

rows = []

for clique_index, (wid, performances) in enumerate(data.items()):
    for version_index, (pid, perf_data) in enumerate(performances.items()):
        rows.append({
            "clique": clique_index,
            "version": version_index,
            "title": perf_data["perf_title"],
            "performer": perf_data["perf_artist"],
            "id": pid 
        })

df = pd.DataFrame(rows)
df.to_csv("datacos_listed.csv", index=False)

# this code was used to make Datacos listed csv from the metadata json