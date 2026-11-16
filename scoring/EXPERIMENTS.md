# Experiment Overview

All three experiments are measured on the same 1,500 hidden frames (150 CAMUS patients, apical two- and four-chamber views). Every method is run through the same evaluator on identical inputs. The visible counterpart (`scores_train.json`) uses 50 validation patients with different prompt and caption wording.

## disease_zero_shot (weight 0.65, primary)

Establishes whether a model's image and text spaces line up with clinical findings well enough to grade unseen frames from free-text prompts it was never trained on. Findings must be told apart from their normal or negated statements: LV systolic dysfunction, LV dilation and LA enlargement, each graded from measurements with clinical guidelines.

- `auc`: ranking quality (the headline number).
- `precision`, `recall`: quality of the yes/no decision the model makes when it compares a positive against a negative prompt.

Reporting all three keeps a method from improving the ranking while degrading its decisions, and the other way round. A method whose ranking is not significantly better than chance is scored as failed in every experiment.

## image_text_retrieval (weight 0.10)

Establishes whether a frame is matched to its own measurement-grounded caption among 500 captions from 50 patients. The captions differ in view, cardiac phase, measured values and findings. Recall@10 is primary and Recall@5 secondary.

## view_knn (weight 0.25, constraint)

Establishes that the visual representation keeps the structure of echo acquisition. A frozen nearest-neighbour vote, using neighbours from other patients only, must recover the apical view. Gains in the two multimodal experiments should not be bought by collapsing the image representation onto the text. F1 is scored and accuracy is a diagnostic.
