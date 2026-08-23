"""
AgriML Disease Classification Package
"""

from src.disease.class_mapping import (
    PV_TO_UNIFIED,
    PD_TO_UNIFIED,
    PV_CLASSES,
    PD_CLASSES,
    PV_CLASS_TO_IDX,
    PV_IDX_TO_CLASS,
    PD_CLASS_TO_IDX,
    PD_IDX_TO_CLASS,
    NUM_PV_CLASSES,
    NUM_PD_CLASSES,
    parse_class,
)
from src.disease.dataset import (
    LeafDiseaseDataset,
    get_transforms,
    create_plantvillage_datasets,
    create_plantdoc_datasets,
)
from src.disease.model import (
    build_model,
    load_stage1_and_swap_head,
    swap_head,
    get_num_params,
    freeze_backbone,
    unfreeze_all,
)
from src.disease.disease_info import DISEASE_DB, get_disease_info
from src.disease.disease_predictor import DiseasePredictor

__all__ = [
    "PV_TO_UNIFIED",
    "PD_TO_UNIFIED",
    "PV_CLASSES",
    "PD_CLASSES",
    "PV_CLASS_TO_IDX",
    "PV_IDX_TO_CLASS",
    "PD_CLASS_TO_IDX",
    "PD_IDX_TO_CLASS",
    "NUM_PV_CLASSES",
    "NUM_PD_CLASSES",
    "parse_class",
    "LeafDiseaseDataset",
    "get_transforms",
    "create_plantvillage_datasets",
    "create_plantdoc_datasets",
    "build_model",
    "load_stage1_and_swap_head",
    "swap_head",
    "get_num_params",
    "freeze_backbone",
    "unfreeze_all",
    "DISEASE_DB",
    "get_disease_info",
    "DiseasePredictor",
]
