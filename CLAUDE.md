# closed_loop_bmode_optimization

## Running things

Use the conda env `cubdl` — it has h5py, which the Field II loader needs:

```
C:/Users/sunbaolin/miniconda3/envs/cubdl/python.exe
```

Long jobs (calibration fits, label generation) are for the user to run, not the
agent. Write the script, hand over the command.

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
