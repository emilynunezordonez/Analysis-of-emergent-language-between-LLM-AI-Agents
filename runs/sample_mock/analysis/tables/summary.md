# Results summary

> **MOCK LLM RUN.** Messages and reconstructions come from the scripted mock backend. These numbers test the pipeline; they are *not* findings about LLM communication.

Analysed text: **received** message. Entropy estimates from small samples are biased (plug-in estimates are too low). Values use the Miller-Madow correction with 95% basic-bootstrap CIs over messages. Conditional (bigram) entropies need far more data than unigrams: check `token_bigram_samples_per_type`. Rows flagged `undersampled` have fewer than 10 tokens per token type and should not be interpreted. Unigram entropy per token is not information per message: also compare `est_bits_per_message_unigram` and compression bytes per message.

| condition | n_messages | mean_tokens | vocab_size | type_token_ratio | H_token_mm | H_token_mm_boot_bc | H_token_mm_ci_low | H_token_mm_ci_high | H_token_cond_mm | H_char_mm | H_char_cond_mm | redundancy_token_observed | redundancy_char_observed | lzma_bits_per_char_net | token_samples_per_type | token_bigram_samples_per_type | undersampled | perf_attribute_score | perf_ssim | perf_exact_match | rx_valid_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_free__budget-inf__mem-off | 40 | 29.225 | 43 | 0.037 | 5.012 | 5.029 | 4.979 | 5.090 | 0.805 | 4.210 | 2.434 | 0.081 | 0.151 | 0.976 | 27.186 | 13.753 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-50__mem-off | 40 | 29.400 | 44 | 0.037 | 5.043 | 5.061 | 5.024 | 5.104 | 0.815 | 4.208 | 2.438 | 0.081 | 0.152 | 0.995 | 26.727 | 13.364 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-20__mem-off | 40 | 12.950 | 36 | 0.069 | 4.778 | 4.825 | 4.638 | 5.078 | 1.180 | 4.362 | 1.952 | 0.085 | 0.159 | 1.412 | 14.389 | 6.727 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-10__mem-off | 40 | 8.650 | 56 | 0.162 | 5.654 | 5.782 | 5.693 | 5.923 | 1.326 | 4.469 | 2.204 | 0.046 | 0.134 | 2.971 | 6.179 | 3.145 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-5__mem-off | 40 | 4.650 | 37 | 0.199 | 4.939 | 5.091 | 4.937 | 5.294 | 1.686 | 3.386 | 2.071 | 0.079 | 0.092 | 4.612 | 5.027 | 2.548 | True | 0.625 | 0.960 | 0.350 | 1.000 |
| natural_language__budget-inf__mem-off | 40 | 24.000 | 29 | 0.030 | 4.651 | 4.662 | 4.644 | 4.688 | 0.716 | 4.215 | 2.028 | 0.047 | 0.150 | 0.720 | 33.103 | 17.143 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| natural_language__budget-50__mem-off | 40 | 24.000 | 29 | 0.030 | 4.651 | 4.662 | 4.643 | 4.685 | 0.716 | 4.215 | 2.028 | 0.047 | 0.150 | 0.720 | 33.103 | 17.143 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| natural_language__budget-20__mem-off | 40 | 18.250 | 29 | 0.040 | 4.658 | 4.673 | 4.647 | 4.711 | 0.712 | 4.186 | 2.012 | 0.047 | 0.156 | 0.874 | 25.172 | 13.273 | False | 0.840 | 0.947 | 0.350 | 1.000 |
| natural_language__budget-10__mem-off | 40 | 10.000 | 20 | 0.050 | 4.092 | 4.111 | 4.079 | 4.153 | 0.653 | 4.074 | 1.779 | 0.061 | 0.135 | 1.081 | 20.000 | 10.256 | False | 0.485 | 0.877 | 0.000 | 1.000 |
| natural_language__budget-5__mem-off | 40 | 5.000 | 8 | 0.040 | 2.850 | 2.861 | 2.845 | 2.905 | 0.323 | 3.660 | 0.682 | 0.058 | 0.088 | 1.099 | 25.000 | 20.000 | False | 0.000 | 0.908 | 0.000 | 0.975 |
| json__budget-inf__mem-off | 40 | 76.700 | 28 | 0.009 | 3.311 | 3.313 | 3.296 | 3.328 | 1.740 | 4.276 | 1.992 | 0.313 | 0.138 | 0.500 | 109.571 | 57.887 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| json__budget-50__mem-off | 40 | 47.550 | 28 | 0.015 | 3.271 | 3.276 | 3.242 | 3.312 | 1.706 | 4.271 | 1.942 | 0.322 | 0.139 | 0.702 | 67.929 | 35.887 | False | 0.522 | 0.907 | 0.350 | 1.000 |
| json__budget-20__mem-off | 40 | 20.000 | 11 | 0.014 | 2.644 | 2.645 | 2.642 | 2.653 | 1.184 | 3.990 | 1.177 | 0.238 | 0.107 | 0.650 | 72.727 | 40.000 | False | 0.310 | 0.870 | 0.000 | 1.000 |
| json__budget-10__mem-off | 40 | 10.000 | 6 | 0.015 | 2.331 | 2.331 | 2.331 | 2.331 | 0.479 | 3.634 | 0.441 | 0.102 | 0.049 | 0.640 | 66.667 | 44.444 | False | 0.000 | 0.908 | 0.000 | 1.000 |
| json__budget-5__mem-off | 40 | 5.000 | 4 | 0.020 | 1.933 | 1.933 | 1.933 | 1.933 | 0.404 | 3.292 | 0.183 | 0.039 | 0.013 | 1.018 | 50.000 | 40.000 | False | 0.000 | 0.908 | 0.000 | 1.000 |
| vector__budget-inf__mem-off | 40 | 27.000 | 9 | 0.008 | 2.503 | 2.508 | 2.471 | 2.561 | 1.254 | 2.503 | 1.254 | 0.212 | 0.212 | 1.630 | 120.000 | 56.842 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| vector__budget-50__mem-off | 40 | 27.000 | 9 | 0.008 | 2.503 | 2.507 | 2.470 | 2.555 | 1.254 | 2.503 | 1.254 | 0.212 | 0.212 | 1.659 | 120.000 | 56.842 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| vector__budget-20__mem-off | 40 | 20.000 | 8 | 0.010 | 2.416 | 2.423 | 2.396 | 2.456 | 1.154 | 2.416 | 1.154 | 0.197 | 0.197 | 2.040 | 100.000 | 50.000 | False | 0.942 | 0.980 | 0.700 | 1.000 |
| vector__budget-10__mem-off | 40 | 10.000 | 7 | 0.018 | 2.399 | 2.407 | 2.372 | 2.447 | 1.015 | 2.399 | 1.015 | 0.149 | 0.149 | 3.040 | 57.143 | 28.571 | False | 0.725 | 0.932 | 0.350 | 1.000 |
| vector__budget-5__mem-off | 40 | 5.000 | 5 | 0.025 | 2.165 | 2.172 | 2.159 | 2.201 | 0.639 | 2.165 | 0.639 | 0.074 | 0.074 | 3.680 | 40.000 | 20.000 | False | 0.396 | 0.866 | 0.000 | 1.000 |
| baseline:template_english | 20 | 24.000 | 29 | 0.060 | 4.672 | 4.693 | 4.665 | 4.738 | 0.736 | 4.220 | 2.039 | 0.047 | 0.150 | 1.250 | 16.552 | 8.571 | False |  |  |  |  |
| baseline:reference_corpus(carroll-alice.txt) | 200 | 30.335 | 1257 | 0.207 | 8.306 | 8.516 | 8.429 | 8.607 | 3.378 | 4.498 | 3.430 | 0.208 | 0.267 | 3.106 | 4.827 | 1.652 | True |  |  |  |  |
