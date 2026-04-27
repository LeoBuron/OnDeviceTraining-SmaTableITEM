# RQ3 — Fine-tuning sample efficiency (new gesture class)

**Question.** When one new gesture class is added, how many samples from how many contributors are required before the new class reaches a stated accuracy floor, while pre-existing classes do not drop below a stated tolerance?

**Sweep axes.**
- samples-per-contributor for the new class
- number of contributors providing those samples

**Measurement.** Two metrics:
- new-class accuracy
- pre-existing-classes accuracy retention (must stay above tolerance)

**Sprint window.** D15 (06.05). Depends on Florian confirming whether a held-out class is acceptable, or whether an unreleased class exists. **First in the cut order after RP2400 + RQ5** if the dataset blocker isn't resolved by D14.

**Layout (planned).**
```
rq3-new-gesture/
├── README.md
├── construction_notes.md  declared dataset construction (load-bearing for the paper since RQ3 manipulates class membership)
├── sweep.yaml
├── run_trial.py
├── analysis.ipynb
└── runs/                  gitignored
```
