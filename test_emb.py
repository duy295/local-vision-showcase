import os, sys
sys.stdout.reconfigure(encoding='utf-8')

emb_dir = r"C:\Users\FPT SHOP\CODING PROBLEM\LLM via FSCIL\my_fscil_data\output_json\output_json\CUB200"
ori_dir = r"E:\DATASET-FSCIL\CUB_200_2011\building model\test"

# --- Embedding files: bỏ _final / _global, bỏ .json ---
emb_raw = os.listdir(emb_dir)
emb_normalized = {}
for f in emb_raw:
    name = f.replace('.json', '').replace('_final', '').replace('_global', '')
    name = name.lower()
    emb_normalized[name] = f  # map normalized -> tên gốc

# --- Original files: bỏ số đầu (vd: 117.) và chuyển về lower ---
ori_raw = os.listdir(ori_dir)
ori_normalized = {}
for f in ori_raw:
    # Bỏ prefix số: "117.Clay_colored_Sparrow" -> "Clay_colored_Sparrow"
    name = f.split('.', 1)[-1] if f[0].isdigit() else f
    name = name.lower()
    ori_normalized[name] = f  # map normalized -> tên gốc

emb_keys = set(emb_normalized.keys())
ori_keys = set(ori_normalized.keys())

print("=== IN EMBEDDING but NOT IN ORIGINAL ===")
diff1 = emb_keys - ori_keys
if diff1:
    for k in sorted(diff1):
        print(f"  EMB file: {emb_normalized[k]}")
else:
    print("  -> All match!")

print("\n=== IN ORIGINAL but NOT IN EMBEDDING ===")
diff2 = ori_keys - emb_keys
if diff2:
    for k in sorted(diff2):
        print(f"  ORI file: {ori_normalized[k]}")
else:
    print("  -> All match!")

print(f"\nTotal embedding (after normalize): {len(emb_keys)}")
print(f"Total original  (after normalize): {len(ori_keys)}")
print(f"Matched: {len(emb_keys & ori_keys)}")