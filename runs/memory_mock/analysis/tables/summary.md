# Results summary

> **MOCK LLM RUN.** Messages and reconstructions come from the scripted mock backend. These numbers test the pipeline; they are *not* findings about LLM communication.

Analysed text: **received** message. Entropy estimates from small samples are biased (plug-in estimates are too low). Values use the Miller-Madow correction with 95% basic-bootstrap CIs over messages. Conditional (bigram) entropies need far more data than unigrams: check `token_bigram_samples_per_type`. Rows flagged `undersampled` have fewer than 10 tokens per token type and should not be interpreted. Unigram entropy per token is not information per message: also compare `est_bits_per_message_unigram` and compression bytes per message.

| condition | n_messages | mean_tokens | vocab_size | type_token_ratio | H_token_mm | H_token_mm_boot_bc | H_token_mm_ci_low | H_token_mm_ci_high | H_token_cond_mm | H_char_mm | H_char_cond_mm | redundancy_token_observed | redundancy_char_observed | lzma_bits_per_char_net | token_samples_per_type | token_bigram_samples_per_type | undersampled | perf_attribute_score | perf_ssim | perf_exact_match | rx_valid_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_free__budget-inf__mem-off | 60 | 29.350 | 44 | 0.025 | 5.023 | 5.034 | 5.000 | 5.075 | 0.804 | 4.212 | 2.435 | 0.083 | 0.150 | 0.806 | 40.023 | 20.011 | False | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-inf__mem-on | 60 | 8.133 | 85 | 0.174 | 5.841 | 6.017 | 5.748 | 6.468 | 1.684 | 4.627 | 2.509 | 0.108 | 0.140 | 3.021 | 5.741 | 2.681 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-10__mem-off | 60 | 8.650 | 56 | 0.108 | 5.616 | 5.697 | 5.637 | 5.789 | 1.287 | 4.460 | 2.180 | 0.046 | 0.134 | 2.243 | 9.268 | 4.718 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_free__budget-10__mem-on | 60 | 6.350 | 56 | 0.147 | 5.029 | 5.155 | 4.856 | 5.512 | 1.841 | 3.985 | 2.028 | 0.152 | 0.209 | 3.613 | 6.804 | 3.229 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline:template_english | 20 | 24.000 | 29 | 0.060 | 4.672 | 4.693 | 4.665 | 4.738 | 0.736 | 4.220 | 2.039 | 0.047 | 0.150 | 1.250 | 16.552 | 8.571 | False |  |  |  |  |
| baseline:reference_corpus(carroll-alice.txt) | 200 | 30.025 | 1238 | 0.206 | 8.289 | 8.492 | 8.404 | 8.588 | 3.357 | 4.499 | 3.429 | 0.208 | 0.266 | 3.119 | 4.851 | 1.667 | True |  |  |  |  |
