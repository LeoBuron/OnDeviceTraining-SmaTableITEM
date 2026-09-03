# RQ3 — Fine-tuning sample efficiency (new gesture class)

**Question.** When one new gesture class is added, how many samples from how many contributors are required before the new class reaches a stated accuracy floor, while pre-existing classes do not drop below a stated tolerance?

**Status (2026-09-03).** Stub — `src/examples/rq3_new_gesture.c` prints `RESULT skipped`. The dataset-construction decision with Florian (hold out one of the six existing classes vs. use an unreleased class) is still open; it is load-bearing because RQ3 manipulates class membership. Third in the cut order (after RP2040 and RQ5).

**Sweep axes.**
- samples-per-contributor for the new class
- number of contributors providing those samples

**Measurement.** Two metrics:
- new-class accuracy
- pre-existing-classes accuracy retention (must stay above tolerance)

**Layout.**
```
rq3-new-gesture/
├── README.md              (this file)
├── construction_notes.md  declared dataset construction (to be written once the class decision is made)
└── runs/                  gitignored

src/examples/rq3_new_gesture.c          the HOST/MCU binary (stub today)
hpc/search_space/rq3_new_gesture.json   Optuna grid (to be written)
```
