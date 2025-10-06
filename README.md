# TFM — Korean Emotion Detection (KOTE + KcELECTRA)
 
> Bilingual README — English 🇬🇧 / Español 🇪🇸
 
---
 
## English
 
### Overview
This repository contains an academic project for **multi‑label emotion detection in Korean texts**. It uses a Keras/TensorFlow pipeline with a **KcELECTRA** backbone and the **KOTE** dataset taxonomy (43 emotions + “no emotion”). It includes:
- A ready‑to‑run **inference CLI** (`predict.py`) with **5‑fold ensemble** and **per‑class thresholding**.
- A Jupyter notebook (`KOTE_keras.ipynb`) with training/evaluation workflow.
- A Conda environment file (`entorno.yml`) tailored for GPU (TensorFlow 2.19).
 
> **Note:** The dataset is **not** included in this repo. Please obtain KOTE separately (links below).
 
### Project structure
- `predict.py` — CLI for inference (ensemble of saved folds + averaged thresholds).
- `KOTE_keras.ipynb` — end‑to‑end notebook (training, evaluation, charts).
- `entorno.yml` — reproducible environment (GPU‑ready).
- `Lotes_de_texto_inferencia.txt` — sample Korean sentences for batch inference.
 
### Requirements
- Conda (Miniconda/Anaconda) with CUDA‑capable GPU recommended.
- Python 3.11 (pinned in `entorno.yml`), TensorFlow 2.19, Transformers.
 
Create and activate the environment:
```bash
conda env create -f entorno.yml
conda activate tf-gpu-clean
python -m pip install -U pip
```
 
### Inference (single text)
```bash
python predict.py   --ckpt-dir checkpoints/20250804_011700   --text "오늘 너무 피곤하고 우울해요..."   --topk 5
```
If `--ckpt-dir` is omitted, `predict.py` will try to **auto‑detect** the most recent run under `checkpoints/` that contains a `thresholds/` folder and at least 3 folds.
 
### Inference (batch file)
Use the provided sample file or your own one‑sentence‑per‑line file:
```bash
python predict.py   --ckpt-dir checkpoints/20250804_011700   --batch-file Lotes_de_texto_inferencia.txt   --topk 5
```
 
### What the script does
- Loads all `.keras` fold checkpoints in `--ckpt-dir`.
- Loads JSON thresholds from `--ckpt-dir/thresholds/` and **averages** them.
- Runs an **ensemble average** of fold predictions.
- Applies per‑class thresholds to obtain active labels.
- Prints a clean CLI report **and** the exact **ChatGPT prompt** used for a baseline comparison.
 
### Labels (Spanish names)
`predict.py` prints predictions mapped to a fixed ordering of 44 labels (43 emotions + “sin emoción”), e.g.: *alegría/entusiasmo, tristeza, ira/enfado, desconfianza/duda, ansiedad/preocupación, …*
 
### Dataset & model references
- **KOTE**: 50k Korean online comments (250k cases) annotated for **43 emotions + NO EMOTION** through crowdsourcing. See the LREC‑COLING 2024 paper and HF dataset card.  
  - Paper (Jeon et al., 2024): https://aclanthology.org/2024.lrec-main.1499/  
  - Dataset (HF): https://huggingface.co/datasets/searle-j/kote
- **KcELECTRA**: Korean ELECTRA model trained on Naver News comments/replies (≈17GB), designed for **user‑generated, noisy text**.  
  - Model card: https://huggingface.co/beomi/KcELECTRA-base
 
### Reproducibility & GPU tips
- The environment pins **TensorFlow 2.19** and `tf-keras`. The script sets `TF_USE_LEGACY_KERAS=1` and enables **memory growth** to avoid OOMs.
- For stable runs, keep `MAX_LENGTH=256` (as in the code) unless you know your VRAM headroom.
 
### License & use
- **Data**: follow KOTE’s license/terms and cite the original paper.
- **Models**: follow the respective model card licenses (KcELECTRA: MIT on HF at the time of writing).
 
### How to cite (examples)
If you use this project or its findings, please cite KOTE and KcELECTRA:
- Jeon, D., Lee, J., & Kim, C. (2024). *User Guide for KOTE: Korean Online That‑gul Emotions Dataset.* LREC‑COLING 2024.
- Beomi (2022–2024). *KcELECTRA-base.* Hugging Face model card.
 
---
 
## Español
 
### Descripción general
Este repositorio reúne un proyecto académico de **clasificación multietiqua** de emociones en textos coreanos. Se basa en **KcELECTRA** con Keras/TensorFlow y utiliza la taxonomía del **dataset KOTE** (43 emociones + “sin emoción”). Incluye:
- **Inferencia por CLI** (`predict.py`) con **ensamblado de 5 folds** y **umbrales por clase**.
- Notebook de **entrenamiento/evaluación** (`KOTE_keras.ipynb`).
- Archivo de entorno **Conda** (`entorno.yml`) listo para GPU.
 
> **Aviso:** El dataset **no** está incluido. Debes obtener KOTE por tu cuenta (enlaces abajo).
 
### Estructura del proyecto
- `predict.py` — inferencia (promedio de folds + umbrales promediados).
- `KOTE_keras.ipynb` — flujo de entrenamiento, evaluación y figuras.
- `entorno.yml` — entorno reproducible (GPU).
- `Lotes_de_texto_inferencia.txt` — frases de ejemplo para inferencia por lotes.
 
### Requisitos
- Conda con GPU compatible (recomendado).
- Python 3.11, TensorFlow 2.19, Transformers.
 
Crear y activar el entorno:
```bash
conda env create -f entorno.yml
conda activate tf-gpu-clean
python -m pip install -U pip
```
 
### Inferencia (texto único)
```bash
python predict.py   --ckpt-dir checkpoints/20250804_011700   --text "오늘 너무 피곤하고 우울해요..."   --topk 5
```
Si omites `--ckpt-dir`, el script intentará **detectar automáticamente** el último run válido bajo `checkpoints/` (con `thresholds/` y ≥3 folds).
 
### Inferencia (archivo por lotes)
```bash
python predict.py   --ckpt-dir checkpoints/20250804_011700   --batch-file Lotes_de_texto_inferencia.txt   --topk 5
```
 
### Qué hace el script
- Carga todos los checkpoints `.keras` del directorio indicado.
- Carga los umbrales JSON y los **promedia** por clase.
- Calcula el **promedio de probabilidades** del ensamble de folds.
- Aplica los **umbrales** para activar etiquetas.
- Muestra un informe legible por consola **y** el **prompt de ChatGPT** utilizado para la comparación de referencia.
 
### Dataset y modelo
- **KOTE**: 50k comentarios (250k casos) anotados con **43 emociones + “no emoción”**.  
  - Paper (Jeon et al., 2024): https://aclanthology.org/2024.lrec-main.1499/  
  - Dataset (HF): https://huggingface.co/datasets/searle-j/kote
- **KcELECTRA**: modelo ELECTRA para coreano entrenado con comentarios/respuestas de Naver (≈17 GB), optimizado para **texto ruidoso generado por usuarios**.  
  - Model card: https://huggingface.co/beomi/KcELECTRA-base
 
### Licencia y uso
- **Datos**: respeta la licencia/condiciones de KOTE y cita el paper original.
- **Modelos**: respeta las licencias de cada modelo.
 
### Cómo citar (ejemplos)
- Jeon, D., Lee, J. y Kim, C. (2024). *User Guide for KOTE: Korean Online That‑gul Emotions Dataset.* LREC‑COLING 2024.
- Beomi (2022–2024). *KcELECTRA-base.* Hugging Face model card.
 
---
 
### Acknowledgements / Agradecimientos
Thanks to the KOTE authors and to the maintainers of KcELECTRA and Hugging Face/Transformers.  
Gracias a los autores de KOTE y a los mantenedores de KcELECTRA y HF/Transformers.
