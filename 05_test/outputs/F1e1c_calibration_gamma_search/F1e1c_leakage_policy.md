# F1e1c Leakage Policy

F1e1c does not use validation. It uses train_inner/calibration only.
Gamma is selected by calibration macro-F1. The selected scaled matrix is locked before any full-train validation evaluation.
