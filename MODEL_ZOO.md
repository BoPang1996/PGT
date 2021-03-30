# Model zoo

## Kinetics

| Method | Backbone | Pretrain | Config | top-1 | top-5 | Checkpoint | Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Slow 36x8 + PGT | R50 | from scratch | Kinetics/SLOW_PROG_36x8_R50.yaml | 75.6 | 92.3 |  | 
| Slow 36x8 + PGT | R101 | from scratch | Kinetics/SLOW_PROG_36x8_R101_50.yaml | 76.9 | 92.8 | |
| SlowFast 36x8 + PGT | R50 | from scratch | Kinetics/SLOWFAST_PROG_76x8_R50.yaml | 76.6 | 92.5 |  | 

## Charades

| Method | Backbone | Pretrain | Config | mAP | Checkpoint | Log |
| --- | --- | --- | --- | --- | --- | --- |
| Slow 76x8 + PGT | R50 | K400 | Charades/SLOW_16x8_R50_K400.yaml | 40.2 | |
| SlowFast + PGT 76x8 | R50 | K400 | Charades/SLOWFAST_PROG_76x8_R50_K400.yaml | 43.8 | | |
| Slow + PGT 76x8 | R101 | K400 | Charades/SLOW_PROG_76x8_R101_K400.yaml | 42.7 |  |
| SlowFast + PGT 76x8 | R101 | K400 | Charades/SLOWFAST_PROG_76x8_R101_K400.yaml | 44.3 |  |  |