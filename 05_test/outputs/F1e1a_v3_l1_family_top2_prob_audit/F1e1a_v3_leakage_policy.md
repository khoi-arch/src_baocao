# F1e1a_v3 Leakage Policy

## Main answer

Using validation L3/top2/probability audit to choose smoothing hyperparameters and then reporting the same validation score as final is validation leakage / tuning-to-val.

It is not train-label leakage, and it is not a bug in inference, but it makes the validation result optimistic because the validation set influenced model design.

## Allowed uses of validation audit

Validation audit is acceptable for:
- diagnosing failure mode
- deciding broad research direction
- explaining why subtype boundary is hard
- generating hypotheses

But if a hyperparameter/matrix is chosen from validation audit, the resulting validation score should be described as model-selection validation, not final unbiased performance.

## Clean ways to avoid leakage

Best:
1. Create internal split from original train:
   train_inner + calibration
2. Train L1 on train_inner
3. Generate calibration predictions
4. Derive smoothing matrix from calibration
5. Retrain final model on original train with locked matrix
6. Evaluate once on original validation or separate test

Better:
- K-fold out-of-fold predictions on the training set, derive matrix from OOF.

Acceptable but weaker:
- Use validation audit only to decide direction.
- Choose a very small fixed candidate before seeing final result.
- Report honestly that validation was used in model selection.

Not clean:
- Tune matrix repeatedly on validation and report the best validation score as final.
