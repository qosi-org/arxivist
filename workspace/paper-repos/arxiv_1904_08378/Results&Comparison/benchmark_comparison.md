# Dynamic Eval Benchmark Comparison

| Dataset | Metric | Static | Dynamic(RMSprop) | Paper | Match |
|---------|--------|--------|------------------|-------|-------|
| enwik8 | bpc | 0.993 | 0.94 | 0.94 | ⏳ |
| text8 | bpc | 1.085 | 1.038 | 1.038 | ⏳ |
| WikiText-103 | ppl | 18.3 | 16.4 | 16.4 | ⏳ |

## Improvement Summary
- enwik8: -5.3% vs static
- text8: -4.3% vs static  
- WikiText-103: -9.4% vs static
