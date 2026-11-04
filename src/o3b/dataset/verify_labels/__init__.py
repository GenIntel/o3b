"""Hand re-annotation of a dataset's object frames, to measure its label quality.

Hypothesis H09 in Every9Dv2: sample objects (``sample``), pack what annotating
them needs without their official pose (``pack``), annotate oriented boxes
blind (``viewer``, saved through ``store``), then score them against the
official labels (``compare``).
"""
