import json

with open(r"C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json\CUB200-json\incre\clip_topk_incre_session1.json", "r") as f:
    data = json.load(f)

results = data.get("results", [data.get("result", data)])
first = results[0] if isinstance(results, list) else results

print("Top-level keys:", list(data.keys()))
print("First item keys:", list(first.keys()))

# Show what the topk data looks like
for key in first:
    val = first[key]
    if isinstance(val, list) and val:
        print(f"\nKey '{key}' (list, first entry):", val[0])
    else:
        print(f"\nKey '{key}':", val)