# EXPERIMENTS.md - Experiments we've tried

## Unsuccessful Experiments

### 1. Enhanced Straight Detection with Lookup Table ❌ FAILED
**Expected**: 3-5% improvement
**Result**: 0.99x slower (regression)
**Cause**: Apple M1's branch predictor handles short loops better than lookup table access

### 2. Parallel Bit Manipulation for Rank Mask Building ⚠️ REVERTED
**Expected**: 2-3% improvement
**Result**: 86% faster component, 0% overall improvement
**Cause**: Amdahl's law - optimized component was only ~10% of total evaluation cost