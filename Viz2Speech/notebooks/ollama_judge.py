"""
LLM-as-a-Judge

Requirements:
- Ollama running locally
- Judge model pulled in Ollama (qwen2.5vl:7b, llava:7b, etc.)
- Python dependencies: ollama, Pillow
"""

import ollama
import json
import csv
import os
import re
import time
from dataclasses import dataclass, field, asdict
from PIL import Image
import io

JUDGE_MODEL  = "qwen2.5vl:7b"
NO_SYSTEM_ROLE = JUDGE_MODEL.startswith("llava")
MAX_RETRIES  = 3
RETRY_DELAY  = 2

IMAGE_MAX_SIZE = 960
IMAGE_QUALITY  = 90

SCORE_MIN = 1.0
SCORE_MAX = 5.0

@dataclass
class CaptionSample:
    image_id:   str
    image_path: str
    validation: str
    zero_shot:  str
    sft:        str
    grpo:       str

@dataclass
class CriterionScore:
    score:     float
    reasoning: str

@dataclass
class MethodEval:
    method:        str
    accuracy:      CriterionScore
    accessibility: CriterionScore
    conciseness:   CriterionScore
    fluency:       CriterionScore

    @property
    def total(self) -> float:
        return (
            self.accuracy.score +
            self.accessibility.score +
            self.conciseness.score +
            self.fluency.score
        ) / 4

@dataclass
class ImageResult:
    image_id: str
    evals:    list[MethodEval] = field(default_factory=list)

def compress_image_to_bytes(image_path: str) -> bytes:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(IMAGE_MAX_SIZE / w, IMAGE_MAX_SIZE / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    return buf.getvalue()

PROMPT = """
You are an expert LLM-as-a-judge evaluator for an Indonesian image captioning system called Viz2Speech.
Its core purpose is to help visually impaired users in Indonesia understand images through
spoken Bahasa Indonesia captions generated via Text-to-Speech (TTS).

You will be given:
1. An image (or its visual context)
2. A reference (ground-truth) validation caption in Bahasa Indonesia
3. A generated caption to evaluate

EVALUATION HIERARCHY & JUDGE BIAS PREVENTION
To prevent judge bias and evaluate fairly, you MUST follow this strict hierarchy of truth:
1. Baseline Truth: Use the Reference Caption as the absolute baseline for identifying the core subject of the image.
2. Visual Verification: Use the Image to verify any additional details the generated caption provides that are not in the reference.
3. No Unfair Penalties: DO NOT penalize the generated caption for including extra details AS LONG AS you can visually verify those details in the image AND they serve the accessibility goal.

Your evaluation must heavily penalize two extremes:
- Cognitive Overload: Captions that are excessively verbose, read every UI element, or list every minor label detail (exhausting for a visually impaired user listening via TTS).
- Over-Sparseness/Inaccuracy: Captions that are too brief and sacrifice critical accuracy (e.g., misidentifying objects just to be short).

Score the generated caption on FOUR criteria using the 1-5 scales below:

CRITERIA DEFINITIONS
1. ACCURACY
   Does the caption correctly identify the main subjects and crucial visual details without hallucinating?
   1 = Completely wrong. Severe hallucinations or critically misidentifies the main subject.
   2 = Poor accuracy. Mostly incorrect or contains major hallucinations.
   3 = Partially correct. Identifies the general subject but misses key details or contains minor misidentifications.
   4 = Mostly accurate. Correctly identifies main subjects and most details with only very minor inaccuracies.
   5 = Fully accurate. Perfect identification of main subjects and crucial details with zero hallucinations.

2. ACCESSIBILITY / UTILITY
   Does this provide the right kind of information for a blind user?
   1 = Useless or harmful. Misleading or causes extreme cognitive overload.
   2 = Very poor utility. Mostly irrelevant or overwhelmingly cluttered.
   3 = Moderate utility. Basic understanding but misses important context or includes unnecessary clutter.
   4 = Good utility. Highly relevant with only slight excess or slight omission.
   5 = Perfect utility. Complete, actionable, and relevant understanding without overwhelming clutter.

3. CONCISENESS & TTS SUITABILITY
   Is the caption the ideal length to be comfortably listened to as spoken audio?
   1 = Unusable for TTS. Exhausting text dump or a single useless word.
   2 = Poorly sized. Highly verbose causing listener fatigue, or far too brief.
   3 = Acceptable. Okay for TTS but noticeably padded or slightly too brief.
   4 = Good length. Comfortable to listen to with minor room for improvement.
   5 = Perfectly balanced. Exactly the right length — concise, focused, every word earns its place.

4. FLUENCY
   Is the Bahasa Indonesia natural, grammatically correct, and easy to understand when spoken aloud?
   1 = Incomprehensible. Completely broken grammar or obvious raw machine-translation artifacts.
   2 = Poor fluency. Frequent grammatical errors or highly unnatural phrasing.
   3 = Acceptable. Understandable but contains awkward phrasing or robotic tone.
   4 = Good fluency. Mostly natural and grammatically correct with only minor awkwardness.
   5 = Perfectly fluent. Natural, native-sounding Indonesian that flows flawlessly as spoken audio.

You MUST respond ONLY with a valid JSON object. No markdown fences, no explanation, no extra text.
{
  "accuracy":      { "score": <float 1.0-5.0>, "reasoning": "<one sentence>" },
  "accessibility": { "score": <float 1.0-5.0>, "reasoning": "<one sentence>" },
  "conciseness":   { "score": <float 1.0-5.0>, "reasoning": "<one sentence>" },
  "fluency":       { "score": <float 1.0-5.0>, "reasoning": "<one sentence>" }
}
""".strip()

def sanitize_raw(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.replace("\u00ad", "").replace("\xad", "")
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    return raw.strip()

def repair_truncated_json(s: str) -> str:
    in_str = False
    escaped = False
    out = []
    for ch in s:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            out.append(ch)
            escaped = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        else:
            out.append(ch)
    result = "".join(out)
    if in_str:
        result += '"'
    result = result.rstrip()
    # Close any missing braces
    opens = result.count("{") - result.count("}")
    result += "}" * max(opens, 0)
    return result


def parse_judge_response(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    raw = sanitize_raw(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Pass 3: attempt repair on the extracted block
        try:
            return json.loads(repair_truncated_json(candidate))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from judge response:\n{raw[:400]}")


def judge_caption(
    image_bytes: bytes,
    validation: str,
    caption: str,
    method_name: str,
) -> MethodEval:
    user_text = (
        f"Reference caption (ground truth):\n{validation}\n\n"
        f"Generated caption ({method_name}):\n{caption}\n\n"
        "Evaluate the generated caption against the four criteria and respond with JSON only."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if NO_SYSTEM_ROLE:
                messages = [
                    {
                        "role":    "user",
                        "content": PROMPT + "\n\n" + user_text,
                        "images":  [image_bytes],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user",   "content": user_text, "images": [image_bytes]},
                ]

            response = ollama.chat(
                model=JUDGE_MODEL,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 2048, "format": "json"},
            )

            raw  = response["message"]["content"].strip()
            data = parse_judge_response(raw)

            def parse(key: str) -> CriterionScore:
                return CriterionScore(
                    score=round(max(SCORE_MIN, min(SCORE_MAX, float(data[key]["score"]))), 2),
                    reasoning=str(data[key]["reasoning"]),
                )

            return MethodEval(
                method=method_name,
                accuracy=parse("accuracy"),
                accessibility=parse("accessibility"),
                conciseness=parse("conciseness"),
                fluency=parse("fluency"),
            )

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print(f"All retries failed for [{method_name}]. Using fallback scores.")
                fallback = CriterionScore(score=0, reasoning="[parse error — judge failed]")
                return MethodEval(
                    method=method_name,
                    accuracy=fallback,
                    accessibility=fallback,
                    conciseness=fallback,
                    fluency=fallback,
                )


def evaluate_sample(sample: CaptionSample) -> ImageResult:
    print(f"Evaluating : {sample.image_id}")

    image_bytes = compress_image_to_bytes(sample.image_path)
    result      = ImageResult(image_id=sample.image_id)

    methods = {
        "Zero-Shot": sample.zero_shot,
        "SFT":       sample.sft,
        "GRPO":      sample.grpo,
    }

    for method_name, caption in methods.items():
        print(f"[{method_name}]", end=" ", flush=True)
        eval_ = judge_caption(image_bytes, sample.validation, caption, method_name)
        result.evals.append(eval_)
        print(
            f"avg={eval_.total:.2f}  "
            f"(acc={eval_.accuracy.score:.2f} "
            f"access={eval_.accessibility.score:.2f} "
            f"con={eval_.conciseness.score:.2f} "
            f"flu={eval_.fluency.score:.2f})"
        )

    return result

def results_to_csv(results: list[ImageResult], path: str) -> None:
    rows = []
    for r in results:
        for e in r.evals:
            rows.append({
                "image_id":      r.image_id,
                "method":        e.method,
                "accuracy":      round(e.accuracy.score, 2),
                "accessibility": round(e.accessibility.score, 2),
                "conciseness":   round(e.conciseness.score, 2),
                "fluency":       round(e.fluency.score, 2),
                "avg_score":     round(e.total, 3),
                "acc_reason":    e.accuracy.reasoning,
                "access_reason": e.accessibility.reasoning,
                "conc_reason":   e.conciseness.reasoning,
                "flu_reason":    e.fluency.reasoning,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {path}")


def print_summary(results: list[ImageResult]) -> None:
    from collections import defaultdict

    totals: dict[str, list[float]]           = defaultdict(list)
    crit:   dict[str, dict[str, list[int]]]  = defaultdict(lambda: defaultdict(list))

    for r in results:
        for e in r.evals:
            totals[e.method].append(e.total)
            crit[e.method]["accuracy"].append(float(e.accuracy.score))
            crit[e.method]["accessibility"].append(float(e.accessibility.score))
            crit[e.method]["conciseness"].append(float(e.conciseness.score))
            crit[e.method]["fluency"].append(float(e.fluency.score))

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    print(f"{'═'*62}")
    print(f"  SUMMARY {JUDGE_MODEL}")
    print(f"  {'Method':<12} {'Acc':>5} {'Access':>7} {'Conc':>6} {'Flu':>5} {'AVG':>6}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*6} {'─'*5} {'─'*6}")

    for method in ["Zero-Shot", "SFT", "GRPO"]:
        if method not in totals:
            continue
        print(
            f"  {method:<12} "
            f"{avg(crit[method]['accuracy']):>5.2f} "
            f"{avg(crit[method]['accessibility']):>7.2f} "
            f"{avg(crit[method]['conciseness']):>6.2f} "
            f"{avg(crit[method]['fluency']):>5.2f} "
            f"{avg(totals[method]):>6.2f}"
        )
    print(f"{'═'*62}")

def main():
    try:
        models  = [m["model"] for m in ollama.list()["models"]]
        matched = any(JUDGE_MODEL in m for m in models)
        if not matched:
            print(f"Model '{JUDGE_MODEL}' not found in Ollama.")
            print(f"Run: ollama pull {JUDGE_MODEL}")
            return
        print(f"Judge model: {JUDGE_MODEL}")
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    samples = [
        CaptionSample(
            image_id   = "id_001",
            image_path = "vizwizz-ds/val/val/VizWiz_val_00000087.jpg",
            validation = "sebungkus daging pinggang babi yang diawetkan, irisan tebal peameal",
            zero_shot  = "Gambar ini menampilkan kemasan daging babi yang telah dibungkus plastik, dengan label putih yang mencantumkan nama produk PORK LOIN CURRED SLICE THICK PEAMEAL, berat bersih 0,406 kg, harga $2,68, dan harga per kilogram $6,59, serta kode barcode dan informasi kontak produsen.",
            sft        = "sebuah kemasan plastik berisi daging babi cincang dengan label putih bertuliskan huruf hitam.",
            grpo       = "kemasan plastik transparan berisi daging asap babi tebal dengan stiker putih dan kuning di atasnya. stiker tersebut memiliki kode batang dan menunjukkan harga 2,68 per kilogram. kemasan tersebut juga menyatakan bahwa daging tersebut harus disimpan dalam keadaan beku.",
        ),
        CaptionSample(
            image_id   = "id_002",
            image_path = "vizwizz-ds/val/val/VizWiz_val_00000105.jpg",
            validation = "Layar komputer meminta untuk memilih driver yang akan diinstal, namun muncul layar kesalahan yang menyatakan tidak ada driver perangkat yang ditemukan.",
            zero_shot  = "Gambar menampilkan layar komputer dengan jendela instalasi Windows yang menunjukkan pesan kesalahan: No device drivers were found. Make sure that the installation media contains the correct drivers, and then click OK. Di bawahnya terdapat opsi Browse dan Rescan, serta kotak centang Hide drivers that are not compatible with hardware on this computer.",
            sft        = "tampilan layar komputer yang menunjukkan instalasi windows dengan pesan peringatan bahwa tidak ada driver yang ditemukan.",
            grpo       = "layar komputer yang menampilkan jendela instalasi windows yang meminta pengguna untuk memilih driver yang akan diinstal dan memberikan peringatan bahwa tidak ada driver yang ditemukan.",
        ),
        CaptionSample(
            image_id   = "id_003",
            image_path = "vizwizz-ds/val/val/VizWiz_val_00000749.jpg",
            validation = "Masalah kualitasnya terlalu parah sehingga konten visual tidak dapat dikenali.",
            zero_shot  = "Gambar ini menampilkan bagian atas layar laptop dengan tampilan antarmuka perangkat lunak Skype, di mana terlihat teks Talesynteser skandes dan Opdel udkloe di atas grafik gelombang suara berwarna hitam dan hijau. Di atas layar, terdapat kamera laptop dengan lubang lensa hitam, dan latar belakang dinding berwarna putih dengan tekstur halus.",
            sft        = "gambar close up layar komputer yang menunjukkan sesuatu yang tampak seperti gelombang suara.",
            grpo       = "tampilan jarak dekat dari sebuah laptop yang sedang menjalankan aplikasi skype dengan grafik suara di layarnya.",
        ),
        CaptionSample(
            image_id   = "id_004",
            image_path = "vizwizz-ds/val/val/VizWiz_val_00001170.jpg",
            validation = "enam kue kecil dengan taburan gula bubuk dan isian",
            zero_shot  = "Gambar ini menampilkan beberapa kue kering berbentuk bulat dan pipih, berwarna kecokelatan dengan permukaan yang halus dan sedikit berkilau, disusun rapi di atas piring logam berkilau dengan tepi berhias, yang diletakkan di atas kertas pembungkus tipis. Di latar belakang, terlihat tumpukan kue cokelat berisi kacang, dan di sebelah kiri, ada potongan kue-kue lain yang tampak seperti kue khas Italia.",
            sft        = "sebuah piring perak berisi kue kering kecil berbentuk bulat yang dilapisi gula bubuk diletakkan di atas meja kaca.",
            grpo       = "kue kering berwarna putih dengan lapisan gula yang dilapisi di atasnya dan berada di dalam wadah perak. terdapat juga kue cokelat chip di atas piala perak di belakang barang barang tersebut. barang barang tersebut ditempatkan di atas permukaan kaca. beberapa barang lainnya berada di sebelah kiri dan kanan barang barang tersebut.",
        ),
        CaptionSample(
            image_id   = "id_005",
            image_path = "vizwizz-ds/val/val/VizWiz_val_00007636.jpg",
            validation = "ruang tamu dengan jam berdiri kayu gelap",
            zero_shot  = "Gambar ini menampilkan sebuah jam dinding klasik berwarna cokelat tua dengan desain elegan, berdiri di tengah ruang tamu yang memiliki dinding berwarna krem, tirai putih, dan pintu putih di sebelah kanan; di depannya terdapat meja kaca dengan barang-barang di atasnya dan sebagian sofa hijau di sudut kanan bawah.",
            sft        = "sebuah jam dinding kayu berdiri di depan pintu putih dan di samping tirai putih.",
            grpo       = "sebuah jam dinding besar berwarna cokelat berada di tengah ruangan di depan dinding kuning dengan pintu di sisi kanan dan tirai di belakangnya. ada juga sofa dan meja kaca dengan laptop di atasnya di bagian depan ruangan.",
        ),
    ]

    all_results: list[ImageResult] = []
    for sample in samples:
        result = evaluate_sample(sample)
        all_results.append(result)

    os.makedirs("outputs", exist_ok=True)

    output_path = JUDGE_MODEL.replace(":", "-").replace(".", "") + "_judge_results.json"
    json_path = f"outputs/{output_path}"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"  ✔ JSON saved → {json_path}")

    results_to_csv(all_results, f"outputs/{output_path.replace('_judge_results.json', '_judge_scores.csv')}")
    print_summary(all_results)

if __name__ == "__main__":
    main()
