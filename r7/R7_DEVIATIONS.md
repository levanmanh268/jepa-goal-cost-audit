# R7 deviations from R7_ARCH_LOCK.md (recorded as they happen)

- D1 (27/10, before any landscape output was read): landscape_analyze.py crashed on its first run because the probe
  stores grids as [n,16,16,C]; added a reshape to [n,256,C] (and flattened goal/proprio) before calling energies.py.
  No change to energies, grids, hypotheses or reading rules. Hash after fix recorded below.
d99d84d3b9dd4fcf65840b20183bec8f0d4f39b03d015e0d704a6b7a8dface94  landscape_analyze.py
- D2 (27/10): PAE refuted by its pre-registered mechanism probe (landscape_train_goals.json: H-M1 false, H-M2 false
  for every beta). Per R7_ARCH_LOCK.md, PAE is not pursued: r7_arch_select.py / r7_arch_eval.py were NOT run, and the
  PAE arm was removed from r7_cem_pilot.py (replaced by the TPE arm under R7_TPE_LOCK.md). Hash of the edited pilot
  runner is recorded in lock_hashes_R7_TPE.txt.
- D3 (27/10, 03:02 UTC): the R7 CPU pilot (ChatGPT R7_NEXT_EXPERIMENT, 4 development starts x 3 arms) stopped after 3/12
  arms (start 0: raw, b1, tpe) when the container restarted. Its question (does the frozen B1 head survive inside CEM?)
  is now asked by R9 on up to 16 fresh R8 starts under R8_ADDENDUM_LOCK.md (b1 arm, same CEM code and settings, plus
  the label-free VELDROP-z arm). The 9 remaining pilot arms are deprioritised behind R9 and will be run only if CPU
  time remains; the 3 completed arms are reported descriptively. r7_pilot_summarize.py labels would be INCOMPLETE.
