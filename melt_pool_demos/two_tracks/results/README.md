# Two-track results

The paper uses the continuous `P/sigma/v` controller only.

- `greedy_1_prevblock_coolseed_continuous.json`: selected 1 mm schedule;
- `greedy_{5,0p5,0p1}_localbox_continuous.json` and
  `greedy_2p5_continuous.json`: other schedules in the block-length table;
- `fullfield/TwoBaselineX50Y1Z1Continuous_*`: nominal reference;
- `fullfield/TwoGreedy1PrevCoolX50Y1Z1Continuous_*`: selected replay;
- the corresponding continuous 5, 2.5, 0.5, and 0.1 mm replay files support
  the block-length statistics.

Optimized-dwell, seed-screening, and tolerance-screening results are outside
the paper scope and are not retained here.
