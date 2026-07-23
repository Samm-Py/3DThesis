# Two-track continuous-control example

This is the paper's smallest raster geometry containing a turnaround and
thermal interaction between neighbouring tracks. The presented controller
modulates only absorbed power, lateral beam width, and scan velocity during a
continuous serpentine scan. No beam-off dwell is introduced.

## Retained paper configuration

- two 10 mm tracks separated by 0.1 mm;
- nominal conditions: 150 W, 3 m/s, and 200 µm lateral beam sigma;
- 1 mm control blocks;
- endpoint width/depth tolerance: 1%;
- cool initialization for the first block after the turnaround;
- previous-block initialization for the remaining blocks on track 2.

The selected schedule is
`results/greedy_1_prevblock_coolseed_continuous.json`, and its matched
50/1/1 µm replay is `TwoGreedy1PrevCoolX50Y1Z1Continuous`.

## Block-length evidence

The continuous block-length sweep retained for Table 2 of the paper is:

| control block | tracked depth (µm) |
|---|---:|
| unoptimized baseline | 69.4 ± 12.4 |
| 5 mm | 65.1 ± 9.4 |
| 2.5 mm | 63.4 ± 7.8 |
| **1 mm** | **62.5 ± 6.6** |
| 0.5 mm | 68.4 ± 10.3 |
| 0.1 mm | 101.6 ± 16.3 |

The 1 mm result is used in the square and triangle studies. The 0.5 and
0.1 mm schedules are retained only as evidence that the controller becomes
unreliable below the melt-pool response length.

## Reproducing the selected case

From this directory:

```bash
MP_BOX_BEHIND=0.001 MP_CASE_DIR=$PWD/cases/snapcase_box \
  python optimize_greedy.py --policy continuous --segment-mm 1 \
  --max-iter 12 --tolerance 0.01 \
  --within-line-warm-start previous-block --turn-seed cool

python verify_schedule.py \
  results/greedy_1_prevblock_coolseed_continuous.json \
  Greedy1PrevCool --policy continuous \
  --resx 50e-6 --resy 1e-6 --resz 1e-6

python make_publication_figures.py
```

## Retained outputs

- `results/greedy_1_prevblock_coolseed_continuous.json`: selected schedule;
- `results/fullfield/TwoGreedy1PrevCoolX50Y1Z1Continuous_*`: selected replay;
- continuous 5, 2.5, 0.5, and 0.1 mm schedules and replay summaries used by
  the block-length table;
- `cases/case_baseline_hires_continuous` and
  `cases/case_greedy1prevcool_hires_continuous`: source cases for the maps;
- `figures/twotrack_base_{maps,controls,traces}.pdf`: paper figures.

Later initialization, tolerance, derivative, and optimized-dwell experiments
were diagnostic only and are not part of the retained paper workflow.
