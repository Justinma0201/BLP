# BEYOND LATENT PERSONALIZATION: INTEGRATING HRV-CONDITIONED PERSONALIZATION VIA LLM FOR INTERPRETABLE STRESS DETECTION

## Expected Structure

TILES-2018 and WESAD share the same pipeline and script organization. The following is the intended layout, including input data, prepared folds, and output directories. Configure script paths to match this layout.

```text
.
|-- README.md
|-- TILES-2018/
|   |-- data/
|   |   |-- heartrate_feature/
|   |   |-- feature_mapping_793.csv
|   |   |-- stressd.csv
|   |   |-- part_one-demographics.csv
|   |   |-- part_two-demographics_timings.csv
|   |   `-- HRV_I5F_Tiles/
|   |       `-- fold-{1..5}/
|   |           |-- df_train_etc_stats_selectnum100.csv
|   |           `-- df_test_etc_stats_selectnum100.csv
|   |-- model/
|   |-- result_analysis/
|   `-- <shared dataset scripts listed below>
|-- WESAD/
|   |-- data/
|   |   |-- WESAD/
|   |   |   `-- S*/
|   |   |       |-- S*.pkl
|   |   |       `-- S*_readme.txt
|   |   `-- 5fold_normalized/
|   |       `-- fold-{1..5}/
|   |           |-- train_normalized.csv
|   |           `-- test_normalized.csv
|   |-- model/
|   |-- result_analysis/
|   `-- <shared dataset scripts listed below>
`-- utlis/
    |-- main_Qwen3_embedding.py
    |-- multisimilarityloss.py
    `-- hist.py
```

Both dataset directories are intended to contain:

```text
create_hrv.py
create_personal_description.py
hrv_four_domain_jsonl.py
inference_four.py
inference_only_hrv_four.py
inference_only_word.py
main.py
main_baseline_rewrite.py
wo_extractor.py
wo_hrv_jsonl.py
wo_interpretation.py
wo_word_jsonl.py
```

Intermediate HRV tables, descriptions, JSONL prompts, and embedding CSVs are stored in each dataset's `data/` directory.

## Script Reference

| Script | Purpose |
| --- | --- |
| `create_hrv.py` | Extract numerical HRV features from physiological recordings. |
| `create_personal_description.py` | Convert participant attributes into baseline descriptions. |
| `hrv_four_domain_jsonl.py` | Build full-model prompts combining participant attributes with each of the four HRV domain. |
| `inference_four.py` | Generate four domain-specific LLM interpretations for the full model. |
| `wo_word_jsonl.py` | Build **w/o Attribute Integration** prompts using HRV only. |
| `inference_only_hrv_four.py` | Generate interpretations for **w/o Attribute Integration**. |
| `wo_hrv_jsonl.py` | Build **w/o HRV Integration** prompts using attributes only, with one record per participant. |
| `inference_only_word.py` | Generate profile-based interpretations for **w/o HRV Integration**. |
| `wo_interpretation.py` | Produce **w/o LLM Interpretation** representations by directly embedding input text. |
| `wo_extractor.py` | Produce **w/o Embedding Extractor** representations using the LLM's final-layer, final-prompt-token hidden state. |
| `main.py` | Train and evaluate the full fusion model and its four-source representation ablations. |
| `main_baseline_rewrite.py` | Train and evaluate **w/o HRV Integration**, using one profile-derived embedding per participant. |
| `utlis/main_Qwen3_embedding.py` | Convert generated interpretations into semantic embedding CSVs. |
| `utlis/multisimilarityloss.py` | Provide the custom attribute-aware multi-similarity loss utility. |
| `utlis/hist.py` | Provide class-distribution and hypergraph utilities. |

## Experimental Settings

Values below reflect the current script defaults.

| Parameter | TILES-2018 | WESAD |
| --- | --- | --- |
| Input mode (`--input_mode`) | `full` | `full` |
| Text sources | 4 × 1024 dimensions | 4 × 1024 dimensions |
| Fusion dimension | 100 | 100 |
| Classifier hidden dimension | 16 | 16 |
| Output classes | 2 | 2 |
| Transformer layers (`--encoder_layers`) | 1 | 1 |
| Attention heads (`--num_heads`) | 2 | 2 |
| Positional encoding (`--posenc`) | 1 | 1 |
| Projection/attention/classifier dropout | 0.3 | 0.3 |
| Transformer feed-forward / dropout | PyTorch defaults | PyTorch defaults |
| Optimizer | Adam | Adam |
| Learning rate (`--lr`) | 0.0005 | 0.0005 |
| Weight decay (`--weight_decay`) | 0.0001 | 0.0001 |
| Epochs (`--epochs`) | 400 | 400 |
| Batch size (`--batch_size`) | 16 participants | 16 sequence chunks |
| Gradient accumulation (`--accum_steps`) | 1 | 1 |
| Sequence length (`--seq_len`) | N/A | 128 windows |
| Window stride (`--window_stride_sec`) | N/A | 0.25 s |
| Contrastive sample cap (`--contrastive_max_windows`) | N/A | 512 windows |
| Random seed (`--seed`) | 123 | 123 |
| Cross-validation folds | 5 | 5 |
| Early stopping (`--earlystop`) | 0 | 0 |
| Early-stopping patience (`--earlystop_limit`) | 40 | 40 |
| Validation fraction when early stopping is enabled | 0.20 of training participants | 0.20 of training participants |
| Validation split seed | 2 | 2 |
| Checkpoint selection when early stopping is enabled | Validation UAR | Validation UAR |

### Loss Parameters

| Parameter | Meaning | TILES-2018 | WESAD |
| --- | --- | --- | --- |
| Classification coefficient | Two-class cross-entropy | 1.0 | 1.0 |
| `--lambda_inter` | Stress-label contrastive term | 0.0 | 0.0 |
| `--lambda_intra` | Participant-and-label contrastive term | 0.5 | 0.5 |
| `--lambda_align` | Paired HRV/text alignment term | 0.0 | 0.0 |
| `--lambda_ortho` | Four-source orthogonality term | 1.0 | 0.5 |
| `--lambda_domain` | Participant-adversarial term | 0.0 | 0.0 |
| `--scale_pos` | Multi-similarity positive scale | 7 | 7 |
| `--scale_neg` | Multi-similarity negative scale | 6 | 6 |

## Full Model Pipeline

```text
Physiological recordings                  Participant attributes
          |                                        |
    create_hrv.py                    create_personal_description.py
          |                                        |
          +-------------------+--------------------+
          |                   |
          |        hrv_four_domain_jsonl.py
          |                   |
          |       Time / Freq / MS_RR / MS_dRR prompts
          |                   |
          |           inference_four.py
          |                   |
          |        Four LLM-generated interpretations
          |                   |
          |      utlis/main_Qwen3_embedding.py
          |                   |
Prepared HRV folds     Four semantic embeddings
          |                   |
          +---------+---------+
                    |
                  main.py
```

Run preprocessing and inference from the relevant dataset directory:

1. Run `create_hrv.py` and `create_personal_description.py`.
2. Prepare the binary labels and numerical training/test folds.
3. Run `hrv_four_domain_jsonl.py` to create four prompt files.
4. Run `inference_four.py` for all four HRV groups.
5. Configure the shared embedding utility's input/output pairs, then run `python ../utlis/main_Qwen3_embedding.py`.
6. Run `python main.py --input_mode full` to train and evaluate the full model.

For WESAD inference, select each group with `--source Time`, `Freq`, `MS_RR`, or `MS_dRR`, and use `--input_dir ./data --output_dir ./data`.

The full-model embedding file pairs are listed below, where `{s}` is `time`, `freq`, `ms_rr`, or `ms_drr`:

| Dataset | Generated text | Embeddings consumed by `main.py` |
| --- | --- | --- |
| TILES-2018 | `data/hrv_{s}_description_full.csv` | `data/hrv_{s}_full_embedding.csv` |
| WESAD | `data/hrv_generated_raw_{s}.csv` | `data/hrv_generated_{s}_embedding.csv` |

Each embedding table contains `record_id` and `emb_0`–`emb_1023`. Training joins the four tables to the numerical records by `record_id`. Evaluation reports accuracy, F1, UAR, sensitivity, specificity, and MCC; prediction CSVs are saved under `result_analysis/`.
