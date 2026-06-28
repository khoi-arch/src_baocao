# F1e1a_v4c Leakage / Cleanliness Policy

v4c does not train and does not use validation.

It reads F1e1a_v4 train_inner/calibration outputs only.

Compared with v4b, v4c removes hand-chosen direction weights
0.55/0.30/0.15. Direction allocation uses only calibration mean probabilities:

    target allocation among other malware classes ∝ mean_prob_other

This is cleaner because no direction-weight hyperparameters are introduced.

Top2 and hard pred rates remain useful diagnostics but are not used to set the
matrix.

Remaining caveat:
F1e1a_v4 reused existing preprocessed train tokens. A fully nested experiment
would rebuild preprocessing/tokenization on train_inner only.
