# Experiment Overview

## toy_disease_zero_shot

Tests whether a system can use aligned image and text representations to classify clinically meaningful disease findings on the fixed test split. This is a scaled version of the headline multimodal diagnosis benchmark.

The scored benefit metrics are AUC, precision, and recall. Reporting all three prevents a future method from improving only the ranking metric while collapsing the clinical decision behavior that the paper reports alongside it.

The scored constraint metric is affirmative-vs-negated caption cosine similarity. Lower values indicate stronger separation between affirmative and negated clinical statements, which is required for clinically useful handling of normal or absent findings.

The experiment can compare any method that writes the expected embedding artifact format. This gives future agents a small but meaningful surface for improving the clinical representation without changing the evaluation protocol.
