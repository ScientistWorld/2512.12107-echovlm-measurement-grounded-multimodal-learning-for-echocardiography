# Problem Definition

## Research Question

Can an echocardiography image–text model learn representations that support clinically grounded interpretation of single echo frames? Each training example pairs an apical echo frame with a measurement-grounded caption: the chamber measurements and ejection fraction for that patient, each stated with its guideline-based interpretation (for example "LVEF 38%, moderately reduced LV systolic function"). Each example also comes with a counterfactual caption in which every finding is negated. A good model places frames close to clinically matching statements, so that it can:

- tell abnormal findings from normal or negated ones (zero-shot, from text prompts it has never seen),
- retrieve the caption that belongs to a frame among many similar captions, and
- keep the visual structure of echo acquisition (apical two- vs four-chamber view) in its image representation.

## Why It Matters

Echo interpretation combines view recognition, quantitative measurement and guideline-based reasoning, and reports are written in language full of negations ("no LA enlargement", "normal LV size"). A model that aligns images with this language can grade findings zero-shot from free-text prompts and transfers to downstream echo tasks.

## Data (public CAMUS, measurement-grounded)

- Source: the public CAMUS release (500 adult patients, apical two- and four-chamber half-sequences from end-diastole to end-systole, expert LV / LA segmentations, biplane LVEF). Licence CC BY-NC-SA 4.0; cite Leclerc et al., IEEE TMI 2019.
- Frames are resampled to a fixed physical scale (1.75 mm per pixel, 112 × 112, apex at top centre), so chamber size in pixels is meaningful. Inputs are grayscale and replicated to 3 channels in [0, 1].
- Findings are graded with ASE chamber-quantification partitions (none / mild / moderate / severe) from the measurements, then binarised (any grade above none is positive):
  - LV systolic dysfunction from the biplane LVEF (sex-specific normal limit),
  - LV dilation from the biplane LV end-diastolic volume (sex-specific partitions),
  - LA enlargement from the four-chamber end-systolic LA area.
- Visible data (`data/downloads/camus_vlm/`, built by `scripts/prepare_data.sh`):
  - `train`: 300 patients, 5 evenly spaced frames per sequence per view, 3,000 image–caption pairs (`train_images.npy`, `train.jsonl` with view, frame, phase, measurements, grades, labels, `caption`, `negated_caption`).
  - `val`: 50 patients, same frame sampling, 500 pairs.
  - `python -m data.prepare_camus --train-frames all` also writes every frame of each sequence (≈ 11.5k highly redundant pairs) if you want it.
- Hidden test: 150 other CAMUS patients, 1,500 frames, owned by the evaluator. The prompts and caption wordings there differ from the visible ones.

## Success Criteria

On the hidden patients, a successful system:
1. ranks frames of patients with each finding above frames of patients without it, using only its own image and text embeddings and hidden positive / negative text prompts (AUC), and makes correct decisions with them (precision, recall);
2. retrieves each frame's own caption among the 500 captions of 50 patients;
3. keeps apical two- vs four-chamber view separable in its frozen image embeddings (k-NN), i.e. gains in (1) and (2) must not come at the expense of the visual representation.

A system whose disease discrimination is not significantly better than chance is scored as failed on everything.
