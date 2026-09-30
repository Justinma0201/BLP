# BEYOND LATENT PERSONALIZATION: INTEGRATING HRV-CONDITIONED PERSONALIZATION VIA LLM FOR INTERPRETABLE STRESS DETECTION

TILES-2018 and WESAD implement the same experimental idea at different temporal resolutions: participant-day records for TILES and physiological windows for WESAD. Their pipelines are conceptually symmetric, but their current scripts are not interchangeable. This README describes the checked-in implementation, including the remaining differences and preparation steps required to reproduce an experiment.

The repository contains research scripts rather than a packaged, end-to-end application. Raw datasets, prepared folds, feature mapping tables for TILES, generated artifacts, checkpoints, dependency lock files, and benchmark results are not included in the current tree.

## Expected structure
```text
.
|-- README.md
|-- TILES-2018/
|   |-- data/
|   |     |-- heartrate_feature/
|   |     |-- feature_mapping_793.csv
|   |     |-- stressed.csv
|   |     |-- part_one-demographics.csv
|   |     |-- part_two-demographics_timings.csv
|   |-- create_hrv.py
|   |-- create_personal_description.py
|   |-- hrv_four_domain_jsonl.py
|   |-- inference_four.py
|   |-- inference_only_word.py
|   |-- main.py
|   |-- main_baseline_rewrite.py
|   |-- wo_extractor.py
|   |-- wo_hrv_jsonl.py
|   |-- wo_interpretation.py
|   `-- wo_word_jsonl.py
|-- WESAD/
|   |-- data/
|   |     |-- WESAD/
|   |-- create_hrv.py
|   |-- create_personal_description.py
|   |-- hrv_four_domain_jsonl.py
|   |-- inference_four.py
|   |-- inference_only_hrv_four.py
|   |-- inference_only_word.py
|   |-- main.py
|   |-- main_baseline_rewrite.py
|   |-- wo_extractor.py
|   |-- wo_hrv_jsonl.py
|   |-- wo_interpretation.py
|   `-- wo_word_jsonl.py
`-- utlis/
    |-- main_Qwen3_embedding.py
    |-- multisimilarityloss.py
    `-- hist.py
```

| Script | Role in the experiment |
| --- | --- |
| `create_hrv.py` | Extract dataset-specific numerical HRV features. |
| `create_personal_description.py` | Convert participant attributes into descriptive text. |
| `hrv_four_domain_jsonl.py` | Build four-domain prompts with HRV and participant descriptions. |
| `inference_four.py` | Generate domain-specific LLM interpretations. |
| `wo_word_jsonl.py` | Build w/o HRV Integration prompts, excluding participant descriptions from the prompt. |
| `inference_only_hrv_four.py` | Generate interpretations from w/o Attribute Integration prompts. |
| `wo_hrv_jsonl.py` | Build w/o Attribute Integration prompts, one record per participant. |
| `inference_only_word.py` | Generate interpretations from w/o HRV Integration prompts. |
| `wo_interpretation.py` | Generate w/o LLM Interpretation Semantic Representation. |
| `wo_extractor.py` | Generate w/o Embedding Extractor Semantic Representation. |
| `main.py` | main. |
| `main_baseline_rewrite.py` | main for w/o HRV Integration. |
| `utlis/main_Qwen3_embedding.py` | Embed generated text into CSV vectors. |
| `utlis/multisimilarityloss.py` | Legacy/custom attribute-aware multi-similarity loss implementation. |
| `utlis/hist.py` | Class-distribution and hypergraph utilities; not instantiated in the active fusion training path. |

## Fusion model and training objective

The main model applies an independent learned projection to each of the four text vectors. A learned, global softmax weight combines the four projected sources into one vector per record. This weight is shared across records; it is not a separate gating network for each sample.

The fused text and numerical HRV vectors are projected to a common width of 100. The HRV sequence passes through optional sinusoidal positional encoding and a Transformer encoder. Bidirectional cross-attention lets the HRV sequence attend to text and text attend to HRV. Residual connections and layer normalization precede concatenation and a two-class prediction head.

Predictions are produced for each valid day/window, not just once per participant. Padding labels use `-1` and are ignored by cross-entropy.

The objective is:

```text
L = L_class
  + lambda_inter  * L_inter
  + lambda_intra  * L_intra
  + lambda_align  * L_align
  + lambda_ortho  * L_ortho
  + lambda_domain * L_domain
```

| Term | Implementation meaning |
| --- | --- |
| `L_class` | Two-class softmax cross-entropy, optionally weighted by inverse class counts. |
| `L_inter` | Multi-similarity loss grouped by stress label across participants and modalities. |
| `L_intra` | Multi-similarity loss grouped by participant ID and stress label; it is not an explicit same-participant-only negative-pair filter. |
| `L_align` | Multi-similarity loss pairing the HRV/text representations of the same record. |
| `L_ortho` | Pairwise cross-correlation penalty between projected text sources. |
| `L_domain` | Participant identity classification through gradient reversal. “Domain” here means participant identity, not TILES versus WESAD. |

Default active auxiliary terms are `lambda_intra = 0.5` and `lambda_ortho = 1.0` for TILES or `0.5` for WESAD. Inter-class, record-alignment, and adversarial-domain terms default to zero. They should not be described as active components of a default run.

`main_baseline_rewrite.py` uses one profile vector and a different projection configuration: a 1024→512→256 text projection and a 256-wide fusion representation. It has no four-source orthogonality term. Consequently, this baseline changes both the text source and model configuration relative to `main.py`.

### Temporal batching and defaults

| Setting | TILES `main.py` | WESAD `main.py` |
| --- | --- | --- |
| Numerical input width | 100 | 61 |
| Text input width | 4 × 1024 | 4 × 1024 |
| Fusion width | 100 | 100 |
| Sequence construction | Participant sequences sorted by date | Contiguous window chunks sorted by timestamp |
| Batch size | 8 participants | 16 chunks |
| Gradient accumulation | 2 | 1 |
| Maximum chunk length | Variable participant sequence length | 128 windows |
| Contrastive subsampling cap | No analogous cap | 512 valid windows |
| Epochs / learning rate | 400 / 0.0005 | 400 / 0.0005 |
| Weight decay / seed | 0.0001 / 123 | 0.0001 / 123 |
| Attention heads / encoder layers | 2 / 1 | 2 / 1 |
| EMA decay | 0.9999 | 0.9999 |
| Early stopping | Disabled by default | Disabled by default |

TILES optionally subsamples days when constructing training batches. WESAD splits chunks at timestamp discontinuities and, when supplied, session/recording/segment boundaries. Its loader accepts the historical mask argument but does not implement TILES-style random day masking.

## Experimental variants

| Variant | Language input or representation | Preparation / training route |
| --- | --- | --- |
| Full model | Four generated interpretations of HRV + profile | `hrv_four_domain_jsonl.py` → `inference_four.py` → embedding utility → `main.py` |
| Without profile text | Four generated interpretations of HRV alone | `wo_word_jsonl.py` → HRV-only inference → embedding utility → `main.py` with corresponding paths |
| Without HRV in the prompt | One generated profile-based description per participant | `wo_hrv_jsonl.py` → `inference_only_word.py` → embedding adaptation → `main_baseline_rewrite.py` |
| Without interpretation | Direct sentence embeddings of input text | `wo_interpretation.py` → configure `main.py` input paths |
| Without sentence extractor | Final prompt-token hidden state from Qwen3.5 | `wo_extractor.py` → adapt embedding width and training inputs |
| Numerical-input ablation | Zero text input at the classifier | `--input_mode hrv_only` |
| Text-input ablation | Zero numerical input at the classifier | `--input_mode text_only` |
| Loss ablation | Disable selected objective terms | Set corresponding `--lambda_*` values to zero |

The `wo_` naming describes a removed component at a specific stage. In particular:

- `wo_word_jsonl.py` removes the personal profile from the LLM prompt; it still produces text from HRV.
- `wo_hrv_jsonl.py` removes HRV from the LLM prompt. Numerical HRV remains in the classifier unless `main_baseline_rewrite.py --input_mode text_only` is used.
- `main.py --input_mode text_only` may still use HRV information indirectly through HRV-derived text.
- Input-mode ablations zero tensors before projections; they do not remove branches, biases, positional encoding, cross-attention, or all auxiliary losses. The loaders still require the embedding files.
- `wo_extractor.py` does not generate an interpretation and then pool its hidden states. It runs a forward pass on the chat-formatted prompt and selects the last token of the last layer, including the assistant generation prefix. Its `--max_new_tokens` argument does not control generation because no generation occurs.

