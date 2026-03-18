# README

## Overview

This project follows a three-stage pipeline:

1. Train the similarity model.
2. Generate or synthesize `Ec` for base and incremental classes.
3. Run testing for each session and fuse the results for final prediction.

---

## 1. Train the similarity model

First, train the similarity model using `train/main.py`.

### Input

* Path to the image dataset
* Required training hyperparameters

### Command

```bash
python train/main.py --data_path <path_to_dataset> [other_arguments]
```

### Output

After training is completed, the model weights will be saved in the `weights/` directory.

---

## 2. Generate `Ec`

After obtaining the trained similarity model, generate `Ec` as follows.

### 2.1. Base classes

For base classes, use:

* `Ec_generate.py`
* Base class data

This step generates `Ec` directly for the base classes using the trained model.

### Command

```bash
python Ec_generate.py [arguments]
```

---

### 2.2. Incremental classes

For incremental classes, `Ec` is synthesized in several steps.

#### Step 1: Find related base classes using CLIP

Use CLIP to determine which base classes can potentially represent each incremental image.

Run:

* `clip_topk_incre_class.py`

This step produces a top-k CLIP file for each image.

### Command

```bash
python clip_topk_incre_class.py [arguments]
```

#### Step 2: Synthesize `Ec`

After obtaining the CLIP top-k results, run:

* `Ec_synthesizer.py`

This step synthesizes `Ec` for incremental classes based on the selected related base classes.

### Command

```bash
python Ec_synthesizer.py [arguments]
```

---

## 3. Testing pipeline

For each session, run the following steps.

### Step 1: Generate CLIP top-k from JSON

Run:

* `clip_topk_from_json_fixed.py`

This script generates the CLIP top-k candidates used for prediction.

### Command

```bash
python clip_topk_from_json_fixed.py [arguments]
```

### Step 2: Predict `Ec` for each class

Run:

* `predict_all.py`

This script generates the `Ec` representation for each class.

### Command

```bash
python predict_all.py [arguments]
```

### Step 3: Fuse results

Run:

* `kan_fusion.py`

This script performs the final fusion step and outputs the final classification result.

### Command

```bash
python kan_fusion.py [arguments]
```

---

## Full pipeline summary

### Training stage

1. Train the similarity model with `train/main.py`.
2. Save trained weights into `weights/`.

### Base class `Ec` generation

1. Run `Ec_generate.py` using base class data.

### Incremental class `Ec` synthesis

1. Use `clip_topk_incre_class.py` to compute top-k related base classes for each image.
2. Use `Ec_synthesizer.py` to synthesize `Ec` for incremental classes.

### Testing stage (for each session)

1. Run `clip_topk_from_json_fixed.py` to generate CLIP top-k candidates.
2. Run `predict_all.py` to generate `Ec` for each class.
3. Run `kan_fusion.py` to obtain the final result.

---

## Notes

* Make sure the similarity model is trained before generating `Ec`.
* Ensure all intermediate JSON or top-k files are generated before moving to the next stage.
* The testing stage must be executed session by session.
