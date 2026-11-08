
# Semi-Supervised Image Classification on CIFAR-10

A KTH group project implementing and evaluating MixMatch and FixMatch against a supervised CNN baseline.

## Experiments

- Varying labeled-data budgets (250–5,000 images)
- FixMatch confidence threshold sensitivity
- Comparison of five strong augmentation strategies

## Results

With 1,000 labeled examples:
- Supervised baseline: 54.56%
- MixMatch: 42.93%
- FixMatch (τ = 0.70): 58.15%

MixMatch was limited by an unsupervised-loss ramp-up that did not complete within the training budget. The methods also used different numbers of training steps, so these results are not compute-matched.

## Report

[Read the full report](./report.pdf)
