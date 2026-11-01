# Reading notes: wide-and-deep CNN for electricity theft

**Paper.** Zheng, Z., Yang, Y., Niu, X., Dai, H.-N. and Zhou, Y. (2018). *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids.* IEEE Transactions on Industrial Informatics, 14(4), 1606–1615.

**What they do.** Daily smart-meter consumption from State Grid Corporation of China (about 42k customers, roughly 1,000 days, with labelled thieves). Two branches trained jointly:
- *Wide* branch: fully connected layer on the raw 1-D daily series (memorises global patterns).
- *Deep* branch: the series is reshaped into a 2-D week × day grid, so a 2-D CNN can see weekly periodicity and compare the same weekday across weeks.
- Preprocessing: missing values interpolated, outliers capped (three-sigma rule), min-max scaling.
- Evaluated with AUC and MAP@k (mean average precision in the top k), and they show the 2-D reshape beats 1-D models.

**What transfers to STEG billing data**

| Paper idea | Our version | Why it differs |
|---|---|---|
| Periodicity via 2-D reshape (week × day) | Conv1D kernel 12 (yearly), stretch: 10 × 12 year × month grid with Conv2D | Our data is monthly bills, not daily reads; the period that matters is the year (seasonal heating/cooling) |
| Interpolate missing days | Explicit **mask channel**, months without an invoice stay 0 | Gaps in billing are themselves a fraud signal (F3); interpolating would hide them |
| Wide branch on raw series | Static branch on the engineered feature store (Dense 64) | Our tabular features (meter status, reconciliation) carry signal a raw series cannot |
| MAP@k | precision@k and recall@k at 1/5/10% budgets | Same idea: only the top of the ranked list gets inspected |
| Three-sigma outlier cap | log1p transform | Consumption is heavy-tailed (99.9th pct ≈ 8,000/month) |

**Caveat.** Their labels come from confirmed theft on a single utility with high-resolution data; our monthly bills spread each invoice evenly over the months it covers, so within-period shape is lost. Expect the CNN to add less here than the paper reports.
