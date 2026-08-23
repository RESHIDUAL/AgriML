"""
class_mapping.py - Canonical source of truth for label mapping.

Maps PlantVillage (38 classes) and PlantDoc (27 classes) folder names to a
shared set of unified canonical labels.  Provides integer encodings for both
Stage 1 (pre-train on PV) and Stage 2 (fine-tune on PD), plus a lightweight
``parse_class`` helper that splits any unified label into its crop, health
status, and disease components.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. PlantVillage folder name → unified canonical label  (38 classes)
# ---------------------------------------------------------------------------
PV_TO_UNIFIED: Dict[str, str] = {
    "Apple___Apple_scab":                                "Apple Scab",
    "Apple___Black_rot":                                 "Apple Black Rot",
    "Apple___Cedar_apple_rust":                          "Apple Cedar Rust",
    "Apple___healthy":                                   "Apple Healthy",
    "Blueberry___healthy":                               "Blueberry Healthy",
    "Cherry_(including_sour)___Powdery_mildew":          "Cherry Powdery Mildew",
    "Cherry_(including_sour)___healthy":                 "Cherry Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":"Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust_":                       "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight":               "Corn Northern Leaf Blight",
    "Corn_(maize)___healthy":                            "Corn Healthy",
    "Grape___Black_rot":                                 "Grape Black Rot",
    "Grape___Esca_(Black_Measles)":                      "Grape Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":        "Grape Leaf Blight",
    "Grape___healthy":                                   "Grape Healthy",
    "Orange___Haunglongbing_(Citrus_greening)":          "Orange Citrus Greening",
    "Peach___Bacterial_spot":                            "Peach Bacterial Spot",
    "Peach___healthy":                                   "Peach Healthy",
    "Pepper,_bell___Bacterial_spot":                     "Bell Pepper Bacterial Spot",
    "Pepper,_bell___healthy":                            "Bell Pepper Healthy",
    "Potato___Early_blight":                             "Potato Early Blight",
    "Potato___Late_blight":                              "Potato Late Blight",
    "Potato___healthy":                                  "Potato Healthy",
    "Raspberry___healthy":                               "Raspberry Healthy",
    "Soybean___healthy":                                 "Soybean Healthy",
    "Squash___Powdery_mildew":                           "Squash Powdery Mildew",
    "Strawberry___Leaf_scorch":                          "Strawberry Leaf Scorch",
    "Strawberry___healthy":                              "Strawberry Healthy",
    "Tomato___Bacterial_spot":                           "Tomato Bacterial Spot",
    "Tomato___Early_blight":                             "Tomato Early Blight",
    "Tomato___Late_blight":                              "Tomato Late Blight",
    "Tomato___Leaf_Mold":                                "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot":                       "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":     "Tomato Spider Mites",
    "Tomato___Target_Spot":                              "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":            "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":                      "Tomato Mosaic Virus",
    "Tomato___healthy":                                  "Tomato Healthy",
}

# ---------------------------------------------------------------------------
# 2. PlantDoc folder name → unified canonical label  (27 classes)
# ---------------------------------------------------------------------------
PD_TO_UNIFIED: Dict[str, str] = {
    "Apple Scab Leaf":            "Apple Scab",
    "Apple leaf":                 "Apple Healthy",
    "Apple rust leaf":            "Apple Cedar Rust",
    "Bell_pepper leaf":           "Bell Pepper Healthy",
    "Bell_pepper leaf spot":      "Bell Pepper Bacterial Spot",
    "Blueberry leaf":             "Blueberry Healthy",
    "Cherry leaf":                "Cherry Healthy",
    "Corn Gray leaf spot":        "Corn Gray Leaf Spot",
    "Corn leaf blight":           "Corn Northern Leaf Blight",
    "Corn rust leaf":             "Corn Common Rust",
    "Peach leaf":                 "Peach Healthy",
    "Potato leaf early blight":   "Potato Early Blight",
    "Potato leaf late blight":    "Potato Late Blight",
    "Raspberry leaf":             "Raspberry Healthy",
    "Soyabean leaf":              "Soybean Healthy",
    "Squash Powdery mildew leaf": "Squash Powdery Mildew",
    "Strawberry leaf":            "Strawberry Healthy",
    "Tomato Early blight leaf":   "Tomato Early Blight",
    "Tomato Septoria leaf spot":  "Tomato Septoria Leaf Spot",
    "Tomato leaf":                "Tomato Healthy",
    "Tomato leaf bacterial spot": "Tomato Bacterial Spot",
    "Tomato leaf late blight":    "Tomato Late Blight",
    "Tomato leaf mosaic virus":   "Tomato Mosaic Virus",
    "Tomato leaf yellow virus":   "Tomato Yellow Leaf Curl Virus",
    "Tomato mold leaf":           "Tomato Leaf Mold",
    "grape leaf":                 "Grape Healthy",
    "grape leaf black rot":       "Grape Black Rot",
}

# ---------------------------------------------------------------------------
# 3 & 4. Sorted lists of unified class names
# ---------------------------------------------------------------------------
PV_CLASSES: List[str] = sorted(set(PV_TO_UNIFIED.values()))
PD_CLASSES: List[str] = sorted(set(PD_TO_UNIFIED.values()))

# ---------------------------------------------------------------------------
# 5. Integer encoding for Stage 1 - PlantVillage (38 classes)
# ---------------------------------------------------------------------------
PV_CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(PV_CLASSES)}
PV_IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in PV_CLASS_TO_IDX.items()}

# ---------------------------------------------------------------------------
# 6. Integer encoding for Stage 2 - PlantDoc (27 classes)
# ---------------------------------------------------------------------------
PD_CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(PD_CLASSES)}
PD_IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in PD_CLASS_TO_IDX.items()}

# ---------------------------------------------------------------------------
# 7 & 8. Convenience constants
# ---------------------------------------------------------------------------
NUM_PV_CLASSES: int = 38
NUM_PD_CLASSES: int = 27

# ---------------------------------------------------------------------------
# 9. Utility - parse a unified label into (crop, is_healthy, disease)
# ---------------------------------------------------------------------------
# Multi-word crop names that must be matched as a unit.
_MULTI_WORD_CROPS = frozenset({
    "Bell Pepper",
})


def parse_class(label: str) -> Tuple[str, bool, str]:
    """Parse a unified canonical label into its semantic components.

    Parameters
    ----------
    label : str
        A unified label such as ``'Tomato Early Blight'`` or
        ``'Bell Pepper Healthy'``.

    Returns
    -------
    tuple[str, bool, str]
        ``(crop_name, is_healthy, disease_name)``

        * *crop_name* - e.g. ``'Tomato'``, ``'Bell Pepper'``
        * *is_healthy* - ``True`` when the disease component is
          ``'Healthy'``
        * *disease_name* - e.g. ``'Early Blight'`` or ``'Healthy'``

    Examples
    --------
    >>> parse_class('Tomato Early Blight')
    ('Tomato', False, 'Early Blight')
    >>> parse_class('Apple Healthy')
    ('Apple', True, 'Healthy')
    >>> parse_class('Bell Pepper Bacterial Spot')
    ('Bell Pepper', False, 'Bacterial Spot')
    """
    # Try multi-word crop names first (longest match).
    for crop in _MULTI_WORD_CROPS:
        if label.startswith(crop + " "):
            disease = label[len(crop) + 1:]
            return crop, disease == "Healthy", disease

    # Default: first token is the crop name.
    parts = label.split(" ", 1)
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else "Healthy"
    return crop, disease == "Healthy", disease


# ---------------------------------------------------------------------------
# 10. Self-test / summary when run directly
# ---------------------------------------------------------------------------
def _main() -> None:
    """Print mapping summaries and run basic sanity checks."""
    print("=" * 65)
    print("AgriML - Class Mapping Summary")
    print("=" * 65)

    # --- Counts ---
    pv_unified = set(PV_TO_UNIFIED.values())
    pd_unified = set(PD_TO_UNIFIED.values())

    print(f"\nPlantVillage folders : {len(PV_TO_UNIFIED):>3}")
    print(f"PlantVillage unified : {len(pv_unified):>3}  (NUM_PV_CLASSES = {NUM_PV_CLASSES})")
    print(f"PlantDoc folders     : {len(PD_TO_UNIFIED):>3}")
    print(f"PlantDoc unified     : {len(pd_unified):>3}  (NUM_PD_CLASSES = {NUM_PD_CLASSES})")
    print(f"Overlap (PD & PV)    : {len(pd_unified & pv_unified):>3}")

    # --- Full mapping tables ---
    print("\n" + "-" * 65)
    print("PlantVillage Mapping  (folder -> unified)")
    print("-" * 65)
    for folder, unified in sorted(PV_TO_UNIFIED.items()):
        idx = PV_CLASS_TO_IDX[unified]
        print(f"  [{idx:>2}] {folder:<55s} -> {unified}")

    print("\n" + "-" * 65)
    print("PlantDoc Mapping  (folder -> unified)")
    print("-" * 65)
    for folder, unified in sorted(PD_TO_UNIFIED.items()):
        idx = PD_CLASS_TO_IDX[unified]
        print(f"  [{idx:>2}] {folder:<35s} -> {unified}")

    # --- Verify every PD unified label exists in PV ---
    print("\n" + "-" * 65)
    print("Verification: every PlantDoc class has a PlantVillage counterpart")
    print("-" * 65)
    missing = pd_unified - pv_unified
    if missing:
        print(f"  [FAIL] {len(missing)} PD class(es) NOT found in PV:")
        for m in sorted(missing):
            print(f"      - {m}")
    else:
        print(f"  [PASS] all {len(pd_unified)} PlantDoc classes exist in PlantVillage.")

    # --- Quick parse_class demo ---
    print("\n" + "-" * 65)
    print("parse_class() examples")
    print("-" * 65)
    examples = ["Tomato Early Blight", "Apple Healthy", "Bell Pepper Bacterial Spot"]
    for ex in examples:
        crop, healthy, disease = parse_class(ex)
        print(f"  '{ex}' -> crop={crop!r}, is_healthy={healthy}, disease={disease!r}")

    # --- Assertion sanity checks ---
    assert len(pv_unified) == NUM_PV_CLASSES, (
        f"Expected {NUM_PV_CLASSES} PV classes, got {len(pv_unified)}"
    )
    assert len(pd_unified) == NUM_PD_CLASSES, (
        f"Expected {NUM_PD_CLASSES} PD classes, got {len(pd_unified)}"
    )
    assert len(PV_CLASSES) == NUM_PV_CLASSES
    assert len(PD_CLASSES) == NUM_PD_CLASSES
    assert not missing, "Some PD classes are missing from PV!"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    _main()
