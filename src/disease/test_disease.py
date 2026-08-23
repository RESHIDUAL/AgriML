import os
import sys
import argparse
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from src.disease.disease_predictor import DiseasePredictor
except ImportError:
    from disease_predictor import DiseasePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriML Leaf Disease Diagnostic CLI"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single leaf image to diagnose",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="Path to a directory of images for batch evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(PROJECT_ROOT, "weights", "leaf_disease_model_final.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_large",
        help="Model backbone architecture",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display",
    )
    return parser.parse_args()


def format_report(image_path: str, result: dict) -> str:
    info = result.get("disease_info", {})
    status_text = "Healthy" if result["is_healthy"] else "Diseased"

    lines = []
    lines.append("=" * 72)
    lines.append(" Leaf Disease Diagnosis Report")
    lines.append("=" * 72)
    lines.append(f"  Image File:    {os.path.basename(image_path)}")
    lines.append(f"  Prediction:    {result['predicted_class']}")
    lines.append(f"  Crop Species:  {result['crop']}")
    lines.append(f"  Health Status: {status_text}")
    lines.append(f"  Condition:     {result['disease']}")
    lines.append(f"  Confidence:    {result['confidence']:.2f}%")
    lines.append("-" * 72)
    lines.append("  Top Predictions:")
    for i, pred in enumerate(result.get("top_k", []), 1):
        lines.append(
            f"    {i}. {pred['class_name']:<38} {pred['confidence']:>6.2f}%"
        )
    lines.append("-" * 72)

    severity = info.get("severity", "N/A")
    lines.append(f"  Severity Index: {severity}")
    symptoms = info.get("symptoms", "N/A")
    lines.append("  Symptoms:")
    for chunk in wrap_text(symptoms, 64):
        lines.append(f"    {chunk}")

    causes = info.get("causes", "N/A")
    if causes and causes != "N/A":
        lines.append("  Pathogen / Causes:")
        for chunk in wrap_text(causes, 64):
            lines.append(f"    {chunk}")

    prevention = info.get("prevention", [])
    if prevention:
        lines.append("  Prevention Protocols:")
        for tip in prevention:
            for chunk in wrap_text(f"- {tip}", 64):
                lines.append(f"    {chunk}")

    chem = info.get("chemical_treatment", [])
    if chem:
        lines.append("  Chemical Controls:")
        for tip in chem:
            for chunk in wrap_text(f"- {tip}", 64):
                lines.append(f"    {chunk}")

    org = info.get("organic_treatment", [])
    if org:
        lines.append("  Organic & Biological Remedies:")
        for tip in org:
            for chunk in wrap_text(f"- {tip}", 64):
                lines.append(f"    {chunk}")

    lines.append("=" * 72)
    return "\n".join(lines)


def wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        if current_len + len(word) + (1 if current else 0) > width:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def main():
    args = parse_args()

    if not args.image and not args.batch_dir:
        print("Error: Specify either --image <path> or --batch-dir <path>.")
        sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at '{args.model}'.")
        sys.exit(1)

    print(f"Loading model from '{model_path}'...")
    predictor = DiseasePredictor(model_path=model_path, backbone=args.backbone)

    if args.image:
        img_path = args.image
        if not os.path.exists(img_path):
            img_path = os.path.join(PROJECT_ROOT, args.image)
        if not os.path.exists(img_path):
            print(f"Error: Image file '{args.image}' not found.")
            sys.exit(1)

        result = predictor.predict(img_path, top_k=args.top_k)
        print(format_report(img_path, result))

    elif args.batch_dir:
        bdir = args.batch_dir
        if not os.path.exists(bdir):
            bdir = os.path.join(PROJECT_ROOT, args.batch_dir)
        if not os.path.isdir(bdir):
            print(f"Error: Directory '{args.batch_dir}' not found.")
            sys.exit(1)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        images = [
            os.path.join(bdir, f)
            for f in sorted(os.listdir(bdir))
            if f.lower().endswith(valid_exts)
        ]

        if not images:
            print(f"No image files found in '{args.batch_dir}'.")
            sys.exit(0)

        print(f"\nEvaluating batch of {len(images)} images from: {args.batch_dir}\n")
        print(f"{'Image File':<32} | {'Predicted Class':<28} | {'Status':<10} | {'Confidence':<10}")
        print("-" * 88)

        for img_path in images:
            try:
                res = predictor.predict(img_path, top_k=1)
                fname = os.path.basename(img_path)
                status_str = "Healthy" if res["is_healthy"] else "Diseased"
                print(
                    f"{fname[:30]:<32} | {res['predicted_class'][:26]:<28} | {status_str:<10} | {res['confidence']:>6.2f}%"
                )
            except Exception as e:
                print(f"{os.path.basename(img_path):<32} | Error: {e}")

        print("-" * 88)


if __name__ == "__main__":
    main()
