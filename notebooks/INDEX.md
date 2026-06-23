# Notebook Index

9 tutorial notebooks, ordered by chapter.
Each notebook focuses on a single conceptual layer.
Downstream notebooks assume upstream outputs/caches exist.

```text
ch2  -> ch3_1 -> ch3_2 -> ch3_3
                 -> ch4_1 -> ch4_2 -> ch4_3
                          -> ch5_1 -> ch5_2
```

| notebook | runtime (quick) | key inputs | outputs | upstream deps | downstream used-by |
| --- | --- | --- | --- | --- | --- |
| `notebooks/chapter2_distribution_transport.ipynb` | `<TBD min>` | Entry point; no upstream notebook inputs. | EB coupling, path-construction, energy-proxy, and solver-in-loop diagnostics in `figures/ch02/fig02_*.png`, `figures/ch02/fig02_*.svg`, and `outputs/ch02/table02_*.csv`. | — | ch3_1 |
| `notebooks/chapter3_1_flow_matching_from_scratch.ipynb` | `<TBD min>` | Chapter 2 distribution-transport background. | Toy CFM endpoint, loss, evolution, conditional-vs-marginal, and object-hierarchy figures in `figures/ch03/fig_toy_*.png`, `figures/ch03/fig_toy_*.pdf`, `figures/ch03/fig03_02_*.png`, `figures/ch03/fig03_02_*.pdf`, `figures/ch03/fig03_03_*.png`, and `figures/ch03/fig03_03_*.pdf`. | ch2 | ch3_2 |
| `notebooks/chapter3_2_eb_flow_matching.ipynb` | `<TBD min>` | Chapter 3.1 toy CFM mechanism and inline training loop. | EB training loss, endpoint-pair, rollout, and Euler-step figures in `figures/ch03/figB1_eb20d_train_val_loss.*`, `figures/ch03/fig03_04_*`, `figures/ch03/fig03_08_*`, and `figures/ch03/fig03_09_*`; audit tables in `tables/ch03/ch03_*.csv`; model cache `outputs/ch03/ch03_eb20d_velocity_mlp_seed42.pt` and config `outputs/ch03/ch03_eb20d_main_config_seed42.json`. | ch3_1 | ch3_3, ch4_1, ch5_1 |
| `notebooks/chapter3_3_eb_ablations.ipynb` | `<TBD min>` | Chapter 3.2 EB main CFM model cache, especially `outputs/ch03/ch03_eb20d_velocity_mlp_seed42.pt`. | Solver, compute-quality, time-sampling, capacity, and straightness diagnostics in `figures/ch03/fig03_10_*`, `figures/ch03/figE1_*`, `figures/ch03/figE2_*`, `figures/ch03/figE3_*`, and `figures/ch03/figE5_*`; CSV, Markdown, and TeX tables in `tables/ch03/table*.csv` and `tables/ch03/paper_table*`. | ch3_2 | ch4_1 |
| `notebooks/chapter4_1_coupling_geometry.ipynb` | `<TBD min>` | Chapter 3.2 EB CFM training and Chapter 3.3 ablation diagnostics. | Coupling, rollout, endpoint-fit, epsilon, reflow, and path-geometry figures in `figures/ch04/fig4_1_*`, `figures/ch04/fig4_2_*`, `figures/ch04/fig4_3_*`, `figures/ch04/fig4_5_*`, and `figures/ch04/fig4_10_*`; metrics and caches in `outputs/ch04/*.csv`, `outputs/ch04/*.json`, and `outputs/ch04/cache/exp*`. | ch3_2, ch3_3 | ch4_2, ch4_3, ch5_1, ch5_2 |
| `notebooks/chapter4_2_state_space_assumptions.ipynb` | `<TBD min>` | Chapter 4.1 coupling-geometry outputs and caches. | Toy branch, representation, EB PC-vs-PHATE, and graph-path diagnostics in `figures/ch04/fig4_2_*`, `figures/ch04/fig4_3_*`, `figures/ch04/fig4_5b_*`, and `figures/ch04/fig4_8*`; diagnostics in `outputs/ch04/table4_*.csv`, `outputs/ch04/exp8_*`, and `outputs/ch04/cache/exp5_toy_plans.npz`. | ch4_1 | ch4_3 |
| `notebooks/chapter4_3_sampling_depth.ipynb` | `<TBD min>` | Chapter 4.1 coupling geometry and Chapter 4.2 state-space assumptions. | Sampling-depth, WFR-FM, stochastic-bridge, and claim-boundary artifacts in `figures/ch04/new3/fig4_11*.png`, `figures/ch04/new3/fig4_11*.pdf`, `figures/ch04/new3/fig4_11*.svg`, `outputs/ch04/table4_6*.csv`, `outputs/ch04/table4_7_biological_assumption_boundary.csv`, `outputs/ch04/tableA_4_3_prior_boundary_audit.csv`, and `outputs/ch04/cache/exp9_*.csv`. | ch4_1, ch4_2 | — |
| `notebooks/chapter5_1_timecourse_suite.ipynb` | `<TBD min>` | Chapter 3.2 EB CFM training and Chapter 4.1 coupling geometry; the CFM training loop is not repeated. | Section 5.1 time-pair, hidden-time, rollout, and velocity-jump figures in `figures/ch05/fig5_1_*.png`; result tables in `tables/ch05/tab_5_1_*.csv`; run summary `outputs/ch05/run_summary.json`. | ch3_2, ch4_1 | ch5_2 |
| `notebooks/chapter5_2_perturbation_sciplex.ipynb` | `<TBD min>` | Chapter 5.1 time-course suite and Chapter 4.1 coupling geometry. | Section 5.2 model, split, metric, and example figures in `figures/ch05/new2/fig_5_2_*.png` and `figures/ch05/new2/fig_5_2_*.pdf`; split table `tables/ch05/tab_5_2_sciplex_splits.csv`; audit, RDKit2D, metrics, and run-summary outputs in `outputs/ch05/real_data_audit.json`, `outputs/ch05/rdkit2d_*`, `outputs/ch05/sciplex_metrics_*.csv`, and `outputs/ch05/run_summary_perturbation_sciplex.json`. | ch5_1, ch4_1 | — |
