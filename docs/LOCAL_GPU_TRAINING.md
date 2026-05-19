# Local GPU Fine-tuning (Windows + RTX 4070 Laptop, 8GB VRAM)

The Colab notebook ([notebooks/finetune_colab.md](../notebooks/finetune_colab.md))
is the path of least friction, but local GPU is faster iteration once setup
is done. With 8GB VRAM you can run PP-OCRv4 mobile_rec fine-tuning
comfortably.

---

## 1. Create a dedicated training venv

The main project venv has CPU-only deps. PaddlePaddle GPU + PaddleOCR is a
~2GB install with numpy 1.x pin — best kept separate.

```powershell
cd e:\OCR_Project
py -3.10 -m venv .venv_train
.venv_train\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Use Python 3.10 for this environment. The main `.venv` may be Python 3.12,
but PaddlePaddle 2.6.x wheels are not a good fit for 3.12 on Windows.

---

## 2. Install PaddlePaddle GPU + PaddleOCR

CUDA 12.7 driver is forward-compatible with the CUDA 12.0 wheels.

```powershell
# Paddle GPU build (CUDA 12.0 wheel)
pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

# PaddleOCR + ONNX exporter + numpy 1.x pin
pip install "numpy<2" paddleocr==2.7.3 paddle2onnx==1.3.1
```

Verify GPU access:

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

You should see `PaddlePaddle is installed successfully!` and a mention of
your GPU. If `run_check` fails with a CUDA error, check that `nvidia-smi`
reports CUDA 12.0 or newer.

On Windows, if training fails with missing DLLs such as `cublasLt64_11.dll`
or `zlibwapi.dll`, install the CUDA runtime wheels and prepend their folders
to `PATH` before running training:

```powershell
pip install nvidia-cublas-cu11==11.11.3.6 `
            nvidia-cudnn-cu11==8.9.5.29 `
            nvidia-cuda-runtime-cu11==11.8.89 `
            nvidia-cuda-nvrtc-cu11==11.8.89

$env:Path="e:\OCR_Project\.venv_train\Lib\site-packages\nvidia\zlib\dll_x64;" +
          "e:\OCR_Project\.venv_train\Lib\site-packages\nvidia\cudnn\bin;" +
          "e:\OCR_Project\.venv_train\Lib\site-packages\nvidia\cublas\bin;" +
          "e:\OCR_Project\.venv_train\Lib\site-packages\nvidia\cuda_runtime\bin;" +
          "e:\OCR_Project\.venv_train\Lib\site-packages\nvidia\cuda_nvrtc\bin;$env:Path"
```

`zlibwapi.dll` is not included in those wheels; put the Windows zlib DLL under
`.venv_train\Lib\site-packages\nvidia\zlib\dll_x64\` if cuDNN asks for it.

---

## 3. Clone PaddleOCR for the training tools

```powershell
git clone --depth 1 -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
```

---

## 4. Download pretrained rec checkpoint

```powershell
mkdir PaddleOCR\pretrained
curl.exe -L -o PaddleOCR\pretrained\ch_PP-OCRv4_rec_train.tar `
  https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
tar -xf PaddleOCR\pretrained\ch_PP-OCRv4_rec_train.tar -C PaddleOCR\pretrained\
```

The older multilingual PP-OCRv4 train tar may be unavailable. The official
Chinese+English PP-OCRv4 rec checkpoint is the current fallback; the recognizer
head is still reset for this project's 417-character dictionary.

---

## 5. Drop the training config

Save the YAML from Section 4 of [notebooks/finetune_colab.md](../notebooks/finetune_colab.md)
to `PaddleOCR\configs\rec\finetune_v0.yml`, but change:

- `data_dir` and `label_file_list` paths from `/content/work/rec_dataset_v0`
  → `e:/OCR_Project/data/rec_dataset_v0` (forward slashes; Paddle accepts both)
- `character_dict_path` → `e:/OCR_Project/data/rec_dataset_v0/ppocr_keys.txt`
- `pretrained_model` → `./pretrained/ch_PP-OCRv4_rec_train/student`

**VRAM tuning for 8GB** — adjust the Train loader section:
```yaml
Global:
  use_amp: true
  amp_level: O2

Train:
  loader:
    batch_size_per_card: 64   # down from 96 to fit in 8GB
    num_workers: 2
```

Without AMP, RTX 4070 Laptop 8GB can fit the model but training is much slower
because VRAM stays near the limit.

---

## 6. Train

```powershell
cd PaddleOCR
python tools\train.py -c configs\rec\finetune_v0.yml
```

Expected throughput on RTX 4070 Laptop with batch 64:
- ~150–200 iterations/min
- 1 epoch (63k crops / 64 batch ≈ 990 iters) ≈ 5–7 min
- 15 epochs ≈ 75–105 min total

Watch the console for `acc: 0.xx` after each eval step. Target:
- Epoch 5: acc ≥ 0.65
- Epoch 10: acc ≥ 0.85
- Epoch 15: acc ≥ 0.90

If VRAM OOMs anyway, drop `batch_size_per_card` further (32) or close
VS Code / Chrome to free the ~1GB they hold.

---

## 7. Export inference model + ONNX

```powershell
python tools\export_model.py `
  -c configs\rec\finetune_v0.yml `
  -o Global.pretrained_model=.\output\v0_rec\best_accuracy `
     Global.save_inference_dir=.\output\v0_rec_infer

paddle2onnx `
  --model_dir .\output\v0_rec_infer `
  --model_filename inference.pdmodel `
  --params_filename inference.pdiparams `
  --save_file .\output\v0_rec.onnx `
  --opset_version 13 `
  --enable_onnx_checker True
```

Copy the resulting files to the project:

```powershell
mkdir e:\OCR_Project\models\v0
copy .\output\v0_rec.onnx e:\OCR_Project\models\v0\rec.onnx
copy ..\data\rec_dataset_v0\ppocr_keys.txt e:\OCR_Project\models\v0\
```

---

## 8. After training: plug into RapidOCR

That step happens in the *main* project venv, not the training venv:

```powershell
cd e:\OCR_Project
.venv\Scripts\Activate.ps1

# RapidOCR can load a custom rec ONNX + dict:
python -c "
from rapidocr_onnxruntime import RapidOCR
engine = RapidOCR(
    rec_model_path='models/v0/rec.onnx',
    rec_keys_path='models/v0/ppocr_keys.txt',
)
print(engine('dataset/demo/images/screen_18851_message_fr.png'))
"
```

Once that sanity check passes, re-run the baseline:

```powershell
.venv\Scripts\python.exe scripts\baseline_eval.py --mode raw
.venv\Scripts\python.exe scripts\baseline_report.py
```

…to measure the fine-tuned model against the same 1,500-image test set.

---

## Tips for iterating

- Use `--train-cap 5000` on `scripts/prepare_rec_data.py` to make a tiny
  sandbox dataset and verify the whole train→export→reload loop in ~10
  minutes before committing to a full 15-epoch run.
- PaddleOCR writes TensorBoard logs to `output/v0_rec/` if you set
  `Global.use_visualdl: true` and `pip install visualdl`.
- If acc plateaus low: usually the char dict has a mismatch — print a
  few labels from `rec_train.txt` and confirm every char is in
  `ppocr_keys.txt`.
