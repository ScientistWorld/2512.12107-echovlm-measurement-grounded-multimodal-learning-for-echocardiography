# Paper's Method (EchoVLM) and this reproduction

## Key Contribution

EchoVLM extends CLIP-style image–text pretraining for echocardiography with two clinically informed objectives, trained on measurement-grounded echo image–caption pairs.

## Approach (paper Sec. 3.2, Eq. 1–3, Algorithm 1)

- `L_CLIP`: symmetric image↔text cross-entropy over the in-batch similarity matrix scaled by a learnable temperature τ.
- `L_view` (view-informed contrastive loss): image–image similarities with self excluded. Images of the same standardised view are positives and all other images in the batch are the denominator. The loss is averaged over each anchor's positives and then over the batch.
- `L_neg` (negation-aware loss): `BCEWithLogits(τ · <z_txt, z_neg>, 0)` between each caption and its negated counterpart, which pushes them apart.
- Total: `L = L_CLIP + λ_view L_view + λ_neg L_neg` with λ_view = 0.5 and λ_neg = 0.1. One learnable τ is shared by all terms.
- Paper settings: ViT-B vision encoder initialised from internal echo weights; CLIP text encoder initialised from EchoCLIP; 112×112 images; AdamW (0.9, 0.999), lr 1e-4, weight decay 0.005 (App. Table 15; the main text says 0.05); global batch 512; cosine decay with 200 linear warm-up steps; 20 epochs.

## This reproduction (`method/`, `baseline/`)

- `method/losses.py` implements the three terms exactly as in Algorithm 1. `method/model.py` builds a ViT-B/16 image encoder and the CLIP text transformer. Text is initialised from the public EchoCLIP weights. The internal echo-pretrained ViT-B is not public, so vision starts from OpenAI CLIP ViT-B/16, with position embeddings interpolated to 7×7 patches for 112×112 input.
- `method/train.py` keeps every paper hyperparameter: lr, weight decay, betas, batch 512, 200 warm-up steps, cosine schedule, 20 epochs, λ's.
- Data scale. CAMUS has 300 training patients. Training uses 5 frames per sequence and view (about 10 pairs per patient, close to the paper's ≈12 per patient), which gives 3,000 pairs. With batch 512 that is 5 steps per epoch and 100 steps over 20 epochs. The paper's 200-step warm-up is therefore never completed: the learning rate rises linearly to 5e-5. Training on every frame of each sequence (≈ 38 near-duplicate pairs per patient) over-fits badly. This setting was selected on the visible validation split; see `HARDENING.md`.
- Table 6 rows: `bash scripts/baseline.sh` (L_CLIP only, self-contained in `baseline/`), `bash scripts/ablations.sh` (+L_view, +L_neg), `bash scripts/method.sh` (full).
- `--augment` adds an optional, size-preserving image augmentation (small rotation and translation, intensity jitter). It is not part of the paper and is off by default.

## Main Claims

The primary claim is that measurement-grounded pretraining with view and negation priors improves zero-shot disease classification (Table 6: AUC 79.6 → 86.5 over CLIP-only) and image–text retrieval, without compromising vision features (view classification 92.9 → 95.1). The paper's secondary claims (transfer to segmentation and landmark detection on other datasets) are not part of this gym.
