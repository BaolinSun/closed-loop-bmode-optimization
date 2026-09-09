# closed_loop_bmode_optimization

## Running things

Use the conda env `cubdl` — it has h5py, which the Field II loader needs:

```
C:/Users/sunbaolin/miniconda3/envs/cubdl/python.exe
```

Run things yourself. Hand a command over to the user only when it would take
roughly half an hour or more — a full label generation, a sweep over every
capture. Anything shorter, just run it and report what came back.

## Script output

**Anything a script prints to the terminal must be ASCII.** The Windows console
renders UTF-8 as mojibake, so Chinese in a `print()` is unreadable. Scripts write
their full report to a `.txt` beside themselves (UTF-8, gitignored) and print the
same text; keep both English. Chinese belongs in docstrings, comments, commit
messages and conversation — none of which go through the console.

## Layout

- `bmode_opt/` — the package
- `tests/measure_*.py` — measurements that answer a question; print a report
- `tests/verify_*.py` — checks that assert an invariant
- `tools_*.py` — pipeline steps that write artefacts
- `data/` — gitignored

## Naming

`BImageMode` 0 is **fundamental** transmit, 1 is **tissue harmonic**. Say
fundamental, not "general" — the console's own label for mode 0 reads as "the
ordinary one", which is not what tells the two apart, and it caused real
confusion. In Chinese: 基波成像 / 谐波成像.
