# Experiment Overview

## toy_disease_zero_shot

Tests whether the reproduced system can use aligned image and text representations to classify clinically meaningful disease findings on the fixed test split. This is the scaled version of the paper's headline multimodal diagnosis benchmark.

The scored metrics are AUC, precision, and recall. Reporting all three prevents a future method from improving only the ranking metric while collapsing the clinical decision behavior that the paper reports alongside it.
