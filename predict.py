import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1") # Necesario para tf-keras
os.environ.setdefault("KERAS_BACKEND", "tensorflow") # Hacemos que keras use TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Evita warnings excesivos de TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0") # 0 para evitar warnings de oneDNN
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false") # Evita warnings de tokenizers
import json, glob, time, argparse
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
import tf_keras as keras
from transformers import AutoTokenizer
from transformers.models.electra.modeling_tf_electra import TFElectraModel

# ───────────────────────────────────────────────
# Configuraciones generales
# ───────────────────────────────────────────────

LABELS_ES = [
    "queja/insatisfacción", "bienvenida/amabilidad", "emoción/admiración", "harto/cansado",
    "gratitud", "tristeza", "ira/enfado", "respeto", "expectativa", "arrogancia/desdén",
    "pena/decepción", "resolución solemne", "desconfianza/duda", "orgullo/satisfacción",
    "comodidad", "interés/curiosidad", "cuidado/afecto", "vergüenza", "miedo/terror",
    "desesperación", "patetismo", "asco/repulsión", "molestia/irritación", "absurdo",
    "sin emoción", "derrota/autoodio", "pereza", "agotamiento", "alegría/entusiasmo",
    "realización", "culpa", "odio", "placer (ternura/belleza)", "desconcierto",
    "estupefacción", "carga/reticencia", "aflicción", "aburrimiento", "compasión",
    "sorpresa", "felicidad", "ansiedad/preocupación", "regocijo", "alivio/confianza"
]

MODEL_NAME = "beomi/KcELECTRA-base"
MAX_LENGTH = 256
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Configuramos TensorFlow para usar un solo hilo y evitar warnings
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# ───────────────────────────────────────────────
# Carga de modelos y thresholds promedios
# ───────────────────────────────────────────────

# Busca el directorio más reciente en "checkpoints/" que tenga thresholds y al menos 3 folds
def find_latest_run_dir(checkpoints_root: str = "checkpoints") -> str | None:
    root = Path(checkpoints_root)
    if not root.exists():
        return None
    candidates = []
    for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        has_thresholds = (p / "thresholds").exists()
        has_some_folds = len(list(p.glob("fold*_beomi_KcELECTRA-base.keras"))) >= 3
        if has_thresholds and has_some_folds:
            candidates.append(str(p))
    return candidates[0] if candidates else None

# Carga todos los modelos de un directorio y los devuelve en una lista
def load_ensemble_models(ckpt_dir: str) -> List[keras.Model]:
    pattern = str(Path(ckpt_dir) / "fold*_beomi_KcELECTRA-base.keras")
    model_paths = sorted(glob.glob(pattern))
    if not model_paths:
        raise FileNotFoundError(f"No se encontraron checkpoints en {ckpt_dir}")
    print(f"Cargando {len(model_paths)} modelos desde: {ckpt_dir}")
    models = []
    for mp in model_paths:
        m = keras.models.load_model(
            mp,
            compile=False,
            safe_mode=False,
            custom_objects={
                "TFElectraModel": TFElectraModel,
                "tf.__operators__.getitem": tf.__operators__.getitem,
                "__operators__.getitem": tf.__operators__.getitem,
            },
        )
        models.append(m)
    return models

# Cargamos todos los thresholds de un directorio, los promediamos y devolvemos el vector resultante
def load_and_average_thresholds(ckpt_dir: str) -> np.ndarray:
    thr_dir = Path(ckpt_dir) / "thresholds"
    thr_paths = sorted(glob.glob(str(thr_dir / "fold*_thresholds.json")))
    if not thr_paths:
        raise FileNotFoundError(f"No se encontraron JSONs de thresholds en {thr_dir}")
    thr_list = []
    num_classes = None
    for tp in thr_paths:
        with open(tp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            items = sorted(((int(k), float(v)) for k, v in d.items()), key=lambda x: x[0])
            vec = np.array([v for _, v in items], dtype=np.float32)
        elif isinstance(d, list):
            vec = np.array([float(x) for x in d], dtype=np.float32)
        else:
            raise TypeError(f"Formato inesperado en {tp}: {type(d)}")
        if num_classes is None:
            num_classes = vec.shape[0]
        elif vec.shape[0] != num_classes:
            raise ValueError(f"Los thresholds de {tp} tienen {vec.shape[0]} clases, pero se esperaban {num_classes}.")
        thr_list.append(vec)
    thr_avg = np.mean(np.stack(thr_list, axis=0), axis=0)
    print(f"Thresholds promedio listos (C={thr_avg.shape[0]})")
    return thr_avg

# ───────────────────────────────────────────────
# Tokenización, predicción y umbrales
# ───────────────────────────────────────────────

# Construye el tokenizador desde el modelo preentrenado
def build_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokeniza un único texto y devuelve un diccionario con input_ids y attention_mask
def tokenize_single(text: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
    return {
        "input_ids": tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=max_length)["input_ids"],
        "attention_mask": tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
    }

# Realiza predicción por ensamble y devuelve el vector de probabilidades promedio
def ensemble_predict_proba(models, inputs) -> np.ndarray:
    probs = []
    for m in models:
        y = m.predict(inputs, verbose=0)
        if isinstance(y, (list, tuple)):
            y = y[0]
        probs.append(np.asarray(y))
    return np.mean(np.stack(probs, axis=0), axis=0)[0]

# Aplica los umbrales a un vector de probabilidades y devuelve una máscara booleana
def apply_thresholds(prob_vec: np.ndarray, thr_vec: np.ndarray) -> np.ndarray:
    return (prob_vec >= thr_vec)

# ───────────────────────────────────────────────
# Formateo de salida y prompt para ChatGPT
# ───────────────────────────────────────────────

# Formatea la salida para CLI y devuelve un string
def format_output(text: str, probs: np.ndarray, thr: np.ndarray, labels: List[str], topk: int = 10) -> str:
    active_mask = apply_thresholds(probs, thr)
    active_idx = np.where(active_mask)[0]
    active_idx = active_idx[np.argsort(-probs[active_idx])]
    active_idx = active_idx[:topk]
    order = np.argsort(-probs)[:topk]
    lines = []
    lines.append("════════════════════════════════════════════════════════")
    lines.append("Texto de entrada")
    lines.append(f"{text}")
    lines.append("────────────────────────────────────────────────────────")
    lines.append(f"Etiquetas activas (prob ≥ umbral, top-{topk})")
    if len(active_idx) > 0:
        for i in active_idx:
            lines.append(f" - [{i:02d}] {labels[i]:>25s} | p={probs[i]:.4f}  thr={thr[i]:.3f}")
    else:
        lines.append(" - Ninguna (ninguna probabilidad supera su umbral)")
    lines.append("────────────────────────────────────────────────────────")
    lines.append(f"Top predicciones por probabilidad (ignora umbrales, top-{topk})")
    for i in order:
        lines.append(f" - [{i:02d}] {labels[i]:>25s} | p={probs[i]:.4f}  (thr={thr[i]:.3f})")
    lines.append("════════════════════════════════════════════════════════")
    return "\n".join(lines)

# Construye el prompt para ChatGPT dada la lista de etiquetas y el texto
def build_chatgpt_prompt(labels: List[str], text: str) -> str:
    labels_str = ", ".join(labels)
    return f"""Eres un sistema experto en análisis de emociones en textos coreanos.
Tu tarea es realizar **clasificación multietiqueta** sobre el siguiente texto,
utilizando **exclusivamente** las etiquetas proporcionadas (no inventes etiquetas nuevas).
Analiza el texto en coreano y selecciona las emociones más probables que refleje.

    Reglas:
- Devuelve entre **1 y 5 etiquetas** como máximo.
- Selecciona únicamente etiquetas que estén claramente presentes.
- Si no hay ninguna emoción aplicable, devuelve una lista vacía `[]`.
- El resultado debe ser un **JSON válido** con una lista de strings, sin explicaciones adicionales.

Lista de etiquetas disponibles:
[{labels_str}]

Texto:
\"\"\"{text}\"\"\""""

# ───────────────────────────────────────────────
# Función main y parseo de argumentos
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inferencia (Ensamble 5 folds + thresholds promedio) para KcELECTRA.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directorio bajo checkpoints/. Si no se pasa, se detecta el más reciente.")
    parser.add_argument("--text", type=str, default=None, help="Texto a clasificar (o por stdin).")
    parser.add_argument("--batch-file", type=str, help="Ruta a un fichero con una frase por línea para procesar en lote.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k para mostrar en ambas secciones.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    ckpt_dir = args.ckpt_dir or find_latest_run_dir("checkpoints")
    if not ckpt_dir:
        raise SystemExit("No se encontró un run válido en 'checkpoints/'.")
    print(f"Usando run: {ckpt_dir}")

    # Cargar modelos, thresholds y tokenizador
    models = load_ensemble_models(ckpt_dir)
    thr_avg = load_and_average_thresholds(ckpt_dir)
    if thr_avg.shape[0] != len(LABELS_ES): # Comprobamos si el número de thresholds coincide con el de etiquetas
        raise SystemExit(f"Inconsistencia: thresholds={thr_avg.shape[0]} y etiquetas={len(LABELS_ES)}.")
    tokenizer = build_tokenizer()

    # Preparar lista de textos a procesar
    texts: List[str] = []
    if args.batch_file:
        # Leer líneas del fichero (omitiendo líneas vacías)
        with open(args.batch_file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif args.text:
        texts = [args.text.strip()]
    else:
        # Leer desde stdin (permite pegar varias líneas y Ctrl+D)
        texts = [ln.strip() for ln in os.sys.stdin.read().splitlines() if ln.strip()]

    if not texts:
        raise SystemExit("No hay textos para procesar. Usa --text, --batch-file o redirige stdin.")

    # Procesar cada texto
    for idx, text in enumerate(texts, 1):
        print(f"\n########### Muestra {idx}/{len(texts)} ###########\n")

        # Tokenizar y predecir
        inputs = tokenize_single(text, tokenizer, max_length=MAX_LENGTH)
        prob_vec = ensemble_predict_proba(models, inputs)

        # Mostrar salida
        print(format_output(text, prob_vec, thr_avg, LABELS_ES, topk=args.topk))

        # Imprimir prompt de ChatGPT
        print("\n\n──────────────── PROMPT PARA CHATGPT ────────────────\n")
        print(build_chatgpt_prompt(LABELS_ES, text))
        print("\n##############################################\n")

    print(f"\nTiempo total: {time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    main()
