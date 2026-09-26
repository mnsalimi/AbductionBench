# Manuscript dataset configuration material

`dataset_configurations.tex` is the LaTeX table for Section 3.4 or its appendix. `section_3_4_conclusion.tex` replaces the bracketed placeholder in Section 3.4. The repository does not contain the manuscript source, so these files are ready to `\input` rather than edits to a non-existent paper `.tex` file.

The table is grounded in the 21 static, five interactive, and one passive datasets of the revised PDF's Table 1, `configs/datasets/`, adapter `documentation()` output in `docs/datasets.md`, and `configs/runs/pilot.yaml`. The 50-record target and seed 20260903 come from `pilot.yaml`. The user confirmed three completed episode runs per record. `configs/runs/episode_trio50.yaml` specifies one episode per record within each execution, while the shared `pilot.yaml` setting specifies three repeats. The table reports the completed three-run count without changing historical run configurations.

Source release commits/tags and license identifiers are not pinned in the repository. The engine checks duplicate sample IDs within a dataset, but there is no demonstrated cross-dataset overlap audit. These fields should be verified and recorded before submission.

The PDF attached to the task still contains conflicting 24-, 27-, and 31-dataset totals; the new table follows its 27-row Table 1 and the explicit 21+5+1 breakdown in the abstract. Its Table 1 caption still says one episode per record and should be changed to three.
