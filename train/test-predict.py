import json
from collections import Counter

with open(r"C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\output_json\cub_test_top1.json", "r") as f:
    data = json.load(f)

mismatches = [f"{item['true_class']} -> {item['pred_top1']}" 
              for item in data['results'] if not item['is_correct_top1']]

print("Top 10 cặp class hay nhầm nhất:")
for pair, count in Counter(mismatches).most_common(10):
    print(f"{pair}: {count} lần")