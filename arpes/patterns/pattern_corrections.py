from arpes.corrections.fermi_edge_corrections import apply_copper_fermi_edge_correction

from .pattern_imports import *


def corrections_from_copper_reference():
    # hypothetical dataset
    scan = simple_load("scan")
    reference = simple_load("copper")

    corrected = apply_copper_fermi_edge_correction(scan, reference)

    # note, if you want to debug the correction in a notebook you can do
