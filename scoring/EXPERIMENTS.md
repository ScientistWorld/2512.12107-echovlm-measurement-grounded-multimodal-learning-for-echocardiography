# Experiment Overview

## toy_disease_zero_shot

Tests whether the reproduced system can use aligned image and text representations to classify clinically meaningful disease findings on the fixed test split. This is the scaled version of the paper's headline multimodal diagnosis benchmark.

## toy_image_text_retrieval

Tests whether visual representations preserve enough clinical and measurement context to retrieve the correct paired caption among other test captions. This measures cross-modal alignment independently of binary disease thresholds.

## toy_negation_constraint

Tests whether positive clinical findings remain distinguishable from normal or negated statements. This guards against systems that align echo images with medical language generally but fail the paper's key negative-finding requirement.
