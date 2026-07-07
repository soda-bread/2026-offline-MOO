# BlueBEAR experiment scripts

These are standalone `.py` versions of the current Exp1-8 notebooks, keeping the original notebook-style top-level calls.

Server repository path used in the scripts:

```text
/rds/projects/w/wangsu-building-automation/Huanbo/2026_new_metric
```

Run separately:

```bash
python bluebear/bluebear_exp1_gpr_rbf.py
python bluebear/bluebear_exp2_gpr_matern.py
python bluebear/bluebear_exp3_autogluon_qr.py
python bluebear/bluebear_exp4_bnn.py
python bluebear/bluebear_exp5_prob_rvea_2022.py
python bluebear/bluebear_exp6_prob_moead_2022.py
python bluebear/bluebear_exp7_tgpr_mo_2023.py
python bluebear/bluebear_exp8_ddmoea_gan_2024.py
```

`bluebear_exp6_prob_moead_2022.py` runs as a single-process script.

Each script prints progress to the terminal in real time and also appends the
same output to a matching log file:

```text
bluebear_exp1_gpr_rbf.log
bluebear_exp2_gpr_matern.log
bluebear_exp3_autogluon_qr.log
bluebear_exp4_bnn.log
bluebear_exp5_prob_rvea_2022.log
bluebear_exp6_prob_moead_2022.log
bluebear_exp7_tgpr_mo_2023.log
bluebear_exp8_ddmoea_gan_2024.log
```

Outputs follow the current shared `src` implementation and are written beside `bluebear/config.yaml`:

```text
result_exp1-8.txt
result.csv
```
