# PaddleOCR Recognition Fine-tuning on Colab (PP-OCRv4 mobile_rec)

This notebook fine-tunes the recognition (rec) model only — the detection
model is generally adequate for our screen layouts. We start from the
multilingual PP-OCRv4 mobile checkpoint and adapt it to our 417-character
22-language dictionary using ~60k synthetic crops.

The whole run targets a Colab T4 GPU and should finish in ~30–60 minutes.
Final output: an ONNX rec model that plugs into the local RapidOCR pipeline.

---

## 0. Local prep (run on your Windows machine, not in Colab)

```powershell
# Build prototype crops + dict (already done if you ran prepare_rec_data.py)
.venv\Scripts\python.exe scripts\prepare_rec_data.py `
    --train-source dataset/train `
    --val-source   dataset/test `
    --output-dir   data/rec_dataset_v0

# Zip the dataset for upload
Compress-Archive -Path data/rec_dataset_v0/* -DestinationPath rec_dataset_v0.zip
```

Upload `rec_dataset_v0.zip` to Google Drive (e.g. `MyDrive/ocr_validator/`).

---

## 1. Colab — Runtime setup

> Runtime > Change runtime type > **T4 GPU** (the free tier is fine).

```python
# verify GPU
!nvidia-smi
```

```python
# mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os, shutil
os.makedirs('/content/work', exist_ok=True)
shutil.copy('/content/drive/MyDrive/ocr_validator/rec_dataset_v0.zip', '/content/work/')
!cd /content/work && unzip -q rec_dataset_v0.zip -d rec_dataset_v0
!ls /content/work/rec_dataset_v0
!wc -l /content/work/rec_dataset_v0/rec_train.txt /content/work/rec_dataset_v0/rec_val.txt
```

---

## 2. Install dependencies

```python
# PaddleOCR pinned to a known-good version that supports PP-OCRv4
!pip install -q paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 paddle2onnx==1.2.4
!pip install -q "numpy<2"  # paddle 2.6 still wants numpy 1.x
```

```python
# clone PaddleOCR for the training tools (we need tools/train.py and configs)
!cd /content/work && git clone --depth 1 -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
```

---

## 3. Download pretrained multilingual rec checkpoint

```python
import os
os.chdir('/content/work/PaddleOCR')

!mkdir -p pretrained
# multilingual PP-OCRv4 mobile rec — already trained on Latin + CJK + Cyrillic + Arabic
!wget -q -O pretrained/multilingual_PP-OCRv4_mobile_rec_train.tar \
   https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/multilingual_PP-OCRv4_mobile_rec_train.tar
!tar xf pretrained/multilingual_PP-OCRv4_mobile_rec_train.tar -C pretrained/
!ls pretrained/multilingual_PP-OCRv4_mobile_rec_train
```

If that URL 404s, use the English/Chinese variant as fallback (its conv layers
still transfer; only the final classifier is reset for our dict):

```python
# fallback
# !wget -q -O pretrained/en_PP-OCRv4_mobile_rec_train.tar \
#   https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_mobile_rec_train.tar
# !tar xf pretrained/en_PP-OCRv4_mobile_rec_train.tar -C pretrained/
```

---

## 4. Custom training config

```python
import yaml, pathlib, textwrap

cfg = textwrap.dedent("""\
Global:
  debug: false
  use_gpu: true
  epoch_num: 15
  log_smooth_window: 50
  print_batch_step: 50
  save_model_dir: ./output/v0_rec
  save_epoch_step: 1
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: ./pretrained/multilingual_PP-OCRv4_mobile_rec_train/best_accuracy
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img:
  character_dict_path: /content/work/rec_dataset_v0/ppocr_keys.txt
  max_text_length: 40
  infer_mode: false
  use_space_char: true
  distributed: false
  save_res_path: ./output/v0_rec/predicts.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 1
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 120
    depth: 2
    hidden_dims: 120
    kernel_size: [1, 3]
    use_guide: true
  Head:
    name: CTCHead
    fc_decay: 0.00001

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /content/work/rec_dataset_v0
    label_file_list:
      - /content/work/rec_dataset_v0/rec_train.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
          max_text_length: 40
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
  loader:
    shuffle: true
    batch_size_per_card: 96
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /content/work/rec_dataset_v0
    label_file_list:
      - /content/work/rec_dataset_v0/rec_val.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4
""")

pathlib.Path('configs/rec/finetune_v0.yml').write_text(cfg)
print('wrote configs/rec/finetune_v0.yml')
```

> Note on the SVTR_LCNet algorithm: PP-OCRv4 mobile uses this architecture.
> The pretrained checkpoint above ships matching weights — only the final
> CTC FC head will be reinitialized because our dict size (417) differs
> from multilingual's. PaddleOCR handles that automatically when it loads
> with a mismatched final-layer size.

---

## 5. Train

```python
!python tools/train.py -c configs/rec/finetune_v0.yml
```

You should see lines like:
```
[2026/05/07 ..] epoch: [1/15], iter: 50, lr: 0.000167, loss: 12.34, ..., acc: 0.0234
```

`acc` is **sequence accuracy** (entire string correct). It will start near 0
because the FC head is fresh, then climb. With our prototype 60k crops
expect:

| Epoch | Approx. val acc |
|---|---|
| 1 | 0.30–0.45 |
| 5 | 0.65–0.80 |
| 10 | 0.85–0.92 |
| 15 | 0.90–0.95 |

If acc plateaus below 0.5 after 5 epochs: lower LR (0.0001) or check
that the dict file matches the labels.

---

## 6. Export to inference model + ONNX

```python
# 1) Export PaddlePaddle inference model
!python tools/export_model.py \
    -c configs/rec/finetune_v0.yml \
    -o Global.pretrained_model=./output/v0_rec/best_accuracy \
       Global.save_inference_dir=./output/v0_rec_infer
```

```python
# 2) Convert to ONNX
!paddle2onnx \
    --model_dir ./output/v0_rec_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./output/v0_rec.onnx \
    --opset_version 13 \
    --enable_onnx_checker True
```

```python
# 3) Sanity test the ONNX
import onnxruntime as ort
sess = ort.InferenceSession('./output/v0_rec.onnx')
print('inputs:', [(i.name, i.shape) for i in sess.get_inputs()])
print('outputs:', [(o.name, o.shape) for o in sess.get_outputs()])
```

---

## 7. Download artifacts back to local

```python
import shutil
shutil.copy('./output/v0_rec.onnx',
            '/content/drive/MyDrive/ocr_validator/v0_rec.onnx')
shutil.copy('/content/work/rec_dataset_v0/ppocr_keys.txt',
            '/content/drive/MyDrive/ocr_validator/v0_rec_keys.txt')
```

Then on the Windows machine, copy from Drive into `models/v0/`:
```
models/v0/
  ├── rec.onnx
  └── ppocr_keys.txt
```

---

## 8. Local plug-in (next step, after the model is back)

We will swap RapidOCR's default rec model for our fine-tuned one and
re-run `scripts/baseline_eval.py --mode raw` to measure the improvement.
That step happens locally, in a follow-up session.
