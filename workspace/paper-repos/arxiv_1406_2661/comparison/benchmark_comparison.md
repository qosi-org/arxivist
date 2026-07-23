\# Benchmark Comparison — Generative Adversarial Nets (arxiv\_1406\_2661)



| Metric | Dataset | Paper Value | Your Result | Deviation | Severity |

|---|---|---|---|---|---|

| Parzen log-likelihood | MNIST | 225 ± 2 | -889.29 ± 11.31 | >100% (sign flip) | Critical |



\*\*Note\*\*: TFD not evaluated (not trained in this run).



\## Root cause analysis

The deviation is large enough that it is very unlikely to be explained by training duration alone.

Most likely contributors, in order of expected impact:



1\. \*\*Training duration\*\* — this run used 5 epochs; the paper's setup trains substantially longer

&#x20;  (exact steps unspecified, but GAN sample quality is known to improve slowly over hundreds of

&#x20;  epochs). This alone could account for a large fraction of the gap.

2\. \*\*Unverified architecture\*\* — G/D hidden sizes (240/1200) were assumed from the SIR, not

&#x20;  confirmed against the paper. If real values differ substantially, capacity mismatch would

&#x20;  suppress both G and D quality.

3\. \*\*Parzen sigma not cross-validated\*\* — sigma=0.2 was used directly rather than tuned on a

&#x20;  validation set as the paper describes. This is a known high-sensitivity parameter for this

&#x20;  metric and could shift results by a large margin on its own.

4\. \*\*Small eval sample (200 test images vs full test set)\*\* — increases variance but is unlikely

&#x20;  to explain a gap this large by itself.

