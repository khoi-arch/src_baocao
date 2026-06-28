# F1e1a_v4 Clean Calibration Policy

## What this script is allowed to do

- Split original train into train_inner and calibration.
- Train a temporary L1 model on train_inner only.
- Use calibration predictions to derive a smoothing matrix.
- Save that matrix as a locked candidate for F1e1b.
- Keep validation untouched for later evaluation.

## What this script must not do

- It must not derive smoothing from validation labels or validation predictions.
- It must not tune parameters on validation.
- It must not claim calibration result as final model performance.
- It must not create fake family labels or fake samples.

## Data leakage status

This avoids the major leakage issue: validation-derived hyperparameter selection.

Caveat:
The dataset tokens/offsets are reused from the existing preprocessing output. A fully nested strict study would rebuild preprocessing/tokenization using train_inner only, then derive calibration predictions. That is more expensive and should be noted if required by the report.

## Next clean step

F1e1b should train on the original full train split with the locked matrix from:

    F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv

Then evaluate once on validation. If F1e1b is bad, do not repeatedly tune using validation without creating a new calibration procedure.
