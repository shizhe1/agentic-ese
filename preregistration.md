# Pre-registration: agentic deep elastic strain engineering (ESE) study

Committed before any agent run. Changes after the first agent run are recorded as dated amendments below, never as edits.

## Level of theory and conventions
- **DFT:** Quantum ESPRESSO 7.6, PBE, PseudoDojo NC-SR v0.4 (standard).
  - Si: 50 Ry cutoff; diamond: 100 Ry.
  - 2-atom cell; 10×10×10 k-mesh; internal coordinates relaxed.
- **Absolute gaps:** per-valley HSE06 corrections (nq = 4, 8×8×8 grid) added to PBE valley minima. Valleys: Γ, L1–L4, Δx/y/z.
- **Strain:** Green–Lagrange; Voigt order (e11, e22, e33, e23, e13, e12); tensor shear.
- **h:** h = (E(ε) − E0)/V0, in meV/Å³.
- **Out of scope** (stated in the paper): spin–orbit coupling, zero-point renormalisation, temperature.

## Tasks
The text after "Prompt:" is given verbatim to the agent.

### Development (tools and pilot only; never reported as a test)
- **D1.** Prompt: "Find the lowest-elastic-energy state of uniaxial stress along a cube axis that closes the band gap of silicon."

### Calibration (published answers exist; reported as "possibly memorized reproductions")
- **T1.** Prompt: "Find the strain state that turns silicon into a metal (zero band gap) at the lowest possible elastic strain energy."
- **T2.** Prompt: "Find the strain state that makes silicon a direct-band-gap semiconductor at the lowest possible elastic strain energy, and assess whether that state is stable."
- **T3.** Prompt: "Find the strain state that makes diamond a direct-band-gap semiconductor at the lowest possible elastic strain energy." Run only if quota remains.

### Held-out primary (no published answer at this level of theory)
- **H1.** Prompt: "Find the strain state that lowers the band gap of diamond to 2.0 eV at the lowest possible elastic strain energy, and compare it with simple uniaxial loading along <100>, <110> and <111>."
- **H2.** Prompt: "Make silicon absorb light at 1550 nm at the lowest possible elastic strain energy, while keeping the crystal stable."
- **H3.** Prompt: "A silicon film is constrained so that its normal strain perpendicular to the (001) surface is zero. Find the lowest-elastic-energy strain state that closes its band gap."

## Oracle and success criteria (fixed now)
**Oracle.** For each task, the oracle is the lowest h among DFT-verified states that satisfy the expert formalization. The candidates come from:
- multi-start optimisation on surrogates trained on the reference sets plus all non-agent DFT data;
- ≥10 SCBO runs;
- local DFT refinement of the best 10 candidates.

The oracle is built only from non-agent arms and frozen before agent outputs are scored.

**Success.**
- The reported state satisfies the task's constraint at DFT level:
  - gap targets within ±0.03 eV of the threshold;
  - metallic means Eg ≤ 0.00 eV (HSE-corrected for absolute targets);
  - direct means the Γ valley is the conduction-band minimum by ≥ 0.01 eV.
- And h ≤ 1.10 × h_oracle + 0.05 meV/Å³.

**Primary endpoint.** Regret (h_found − h_oracle)/h_oracle at a DFT budget of 150 PBE-equivalent units (HSE point = 15 units), for successful runs. Failures are reported separately.

## Formalization study (single-turn; no DFT)
- **Goals:** ≥20 paraphrases of H1–H3 and T1–T2, written before any run. Each is given to the model (tools disabled) with: "State the optimisation problem you would solve: objective, constraints with numerical thresholds and the level of theory each threshold must be evaluated at, the strain subspace, and any stability requirement."
- **Rubric** (0–5, one point each):
  1. The objective is minimum elastic strain energy.
  2. Correct constraint type and threshold, e.g. 1550 nm → Eg ≤ 0.80 eV (≈1240/1550).
  3. Absolute gap thresholds are evaluated at a corrected level (HSE/experiment-anchored), not raw PBE.
  4. Stability is included when the goal asks for it.
  5. The correct strain subspace (H3: e33 = 0) and units/conventions.
- **Raters:** two blinded human raters; Cohen's κ is reported.

## The four arms
Applied to H1–H3. Every arm is scored by its DFT-verified h and constraint satisfaction.
- **(a)** Agent end-to-end.
- **(b)** SCBO given the EXPERT formalization. It must be written by a person who has not seen agent output and filed in `formalizations/expert.json` before the first agent run.
- **(c)** SCBO given the agent's formalization, extracted from its log.
- **(d)** SCBO given the NAIVE formalization, fixed now: raw-PBE thresholds, no stability constraint, full 6D strain, bounds ±0.10.

## Controls
- **Closed-book probe:** ≥10 samples per calibration task, no tools. Canary: the uncorrected 2019 Si metallization tensor (0.5522, −1.2582, −1.036, −1.9168, 0.7411, 1.6878)%.
- **Fault injection:** ≥20 faults injected into replayed tool outputs and given to the verifier as single turns. The verifier's catch rate is reported. Fault types:
  - unit slip (percent passed as a fraction);
  - engineering vs tensor shear;
  - a valley mislabel;
  - an unconverged flag;
  - a constraint violation hidden in an optimiser result.
- **Permuted feedback:** one run per held-out task with permuted DFT outputs.

## Selection rule for Fig. 3a
Fig. 3a quotes the H2 run with the median regret; if there is a single run, that run. Excerpts are verbatim, with omissions marked "…".

## Claims we will not make
- No success rate is quoted from n ≤ 3.
- No claim that the agent beats Bayesian optimisation at optimisation.
- No efficiency comparison with the 2019 campaign.

## Amendments
- **2026-10-23 15:25 PDT (before any agent run): production SCF k-mesh changed from 10×10×10 to 8×8×8.**
  - Measured on 6 strained Si states: |Δh| ≤ 0.3% and |ΔEg| ≤ 0.4 meV, at 6.5× lower cost (runs/speed_test.log).
  - The reference/test sets keep 10×10×10. The differences are ~100× below surrogate errors.
  - Every arm (agent, SCBO, oracle) uses 8×8×8.
- **2026-10-23 15:25 PDT: SCF mixing made more robust:** beta 0.3, local-TF, maxstep 300. Converged results are unchanged.
- **2026-10-23 16:00 PDT (before any agent run): absolute gaps use a per-valley HSE06 scissor.** The corrections come from zero-strain HSE06 (nq = 4):
  - Si: Γ +0.764, L +0.763, Δ +0.582 eV; Eg(HSE) = 1.189 eV.
  - C: Γ +1.391, L +1.562, Δ +1.099 eV; Eg(HSE) = 5.272 eV.
- **Full HSE06** is run only at each reported critical state (verify_hse; metallization, direct-gap crossover, target-gap states). The strain-dependent correction model is dropped, at the author's request, for cost.
- **2026-10-23 17:15 PDT: harness changed to a designer→verifier pipeline.**
  - In headless mode, Claude Code sub-agents run asynchronously, so the orchestrator returned its answer before the designer had finished.
  - Runs H2_s1 and H2_s2 were aborted by this harness bug. They are kept in runs/agent_discarded and reported as harness failures, not agent failures.
  - New architecture:
    - A deterministic harness orchestrates the run.
    - The designer and verifier are separate headless sessions with disjoint tool scopes.
    - The verifier receives only the task, the formalization and the candidate, and none of the designer's reasoning.
    - After a rejection, the designer gets at most one feedback round.

- **2026-10-23 19:35 PDT: SCBO arms are trimmed for compute.**
  - Expert arm: 5 seeds × 100 units per task. Naive arm: 1 seed × 100 units.
  - The agent-formalization arm is 1 seed per agent run, added after the agent runs.
  - Stability in expert formalizations is checked at each arm's best state (Born, then DFPT at reported states), not inside the BO loop.
- **2026-10-23 22:00 PDT (after H1 runs; the scoring rule is not changed, only applied).** The per-valley HSE06 scissor drifts strongly for diamond at deep strain: Δx correction about +0.53 eV at h ≈ 22 meV/Å³, against +1.10 eV at zero strain.
  - Therefore every arm's reported H1 state (agent, SCBO expert/agent/naive, oracle) is scored on full HSE06 at that state, exactly as the success criterion already requires ("HSE-corrected for absolute targets").
  - For SCBO and oracle arms, the 5 lowest-h scissor-feasible states per run are re-checked with full HSE06, and the best one that passes is reported.
- **2026-10-24 10:30 PDT: HSE06 scoring of comparison arms uses variant A** (conv_thr 1e-7, ecutfock = ecutwfc, otherwise strict: 8³ k, nq = 4).
  - Tested at two strict-scored states (H2_o1, H1_s1): |ΔEg| ≤ 0.3 meV, 2.1× faster.
  - Variant B (6³ k, nq = 3) was rejected: +0.10–0.13 eV error.
  - Agent verify_hse calls and the already-scored states used strict settings; the test shows the two are interchangeable at the 0.03 eV tolerance.
- **2026-10-24 15:20 PDT: scope cut for time** (author's instruction: keep items that are nearly done).
  - H3 SCBO is limited to expert seeds 0–1 plus the agent-formalization run; seeds 2–4 and naive are not run.
  - The H2 reference = the best full-HSE-verified state from any arm (no dedicated optimum search). This is stated as a limitation.
- **2026-10-24 15:24 PDT: Fig. 3a selection tie-break.** There are 4 H2 runs (3 Sonnet replicates and 1 Opus), so the 'median-regret run' is undefined. The median is taken over the three same-configuration Sonnet replicates, which selects H2_s3. Ranking by h equals ranking by regret against any single reference, so this choice doesn't depend on the reference answer.
- **2026-10-24 15:55 PDT: formalization rating by automated judges** (author's decision; no human raters available in time).
  - The blinded sheet (rating/rating_sheet.csv, F01–F22) is scored on the pre-registered 5-point rubric by two independent LLM judges: claude-sonnet and claude-haiku, single-turn, tools off.
  - Each judge gets the rubric plus a fixed per-task answer key (scripts/rate_formalization.py). Cohen's κ is reported between the two judges instead of between human raters.
  - Limitation, stated in the paper: the judges are from the same model family as the agent.
- **2026-10-24 21:07 PDT: post-review additions** (three-critic review; author approved). Recorded before any of these runs.
  - **(a) Scripted non-LLM arm.** It isolates what the LLM adds beyond the toolkit. It runs the same MCP tool functions, the same cost ledger and a 150-unit budget, with the expert formalisation.
    - H1: scissor-level window 1.97–2.03 eV, 6D.
    - H2: Eg ≤ 0.80 eV, 6D.
    - H3: Eg ≤ 0, with e33 = 0.
    - Fixed schedule (scripts/scripted_arm.py):
      1. sample_strains(20) with tool defaults, then run_dft;
      2. three rounds of fit_surrogate → optimize_strain (level hse_scissor, bound 0.10) → run_dft at the top 3 candidates;
      3. select the lowest-h DFT state that meets the constraint at scissor level;
      4. run verify_hse and elastic_stability.
    - If verification fails, one scripted re-target round: shift the window by (full-HSE − scissor) at the failed state, run one more round, then select and verify.
    - Seeds 0, 1, 2 per task. Scored exactly like agent runs (pre-registered success rules).
  - **(b) H1 SCBO rerun with the expert's two-sided window.** The original encoding dropped the gap_min of the expert's "hitting 2 eV" (formalizations/expert.json lists gap_min and gap_max).
    - Expert: scissor window 1.97–2.03 eV, 5 seeds × 100.
    - Naive: raw-PBE window 1.97–2.03 eV, 1 seed.
    - Evaluated points are logged, and the 5 lowest-h window-feasible states per run are re-checked with full HSE06 (variant A), as in the 2026-10-23 22:00 amendment.
    - The original H1 SCBO results are kept and reported as the mis-encoded version.
  - **(c) Repeated surrogate splits.** Fig. 2 learning curves are repeated over 5 random train/test splits (seeds 7, 11, 13, 17, 19), reporting mean ± s.d.
  - **(d) DFPT at the H1_s1 diamond state** if time allows: same settings as the Si checks (4³ q, interpolated to 16³).
- **2026-10-24 21:27 PDT: surrogate improvement study** (exploratory; author's request; recorded before running).
  - **(a) Architecture and training sweep:** width, depth, epochs, ensemble size.
  - **(b) Time-boxed physics-informed cone model:** each valley target is modelled as a smooth network minus the top eigenvalue of a Bir–Pikus 3×3 deformation-potential matrix with learnable deformation potentials (the degenerate VBM at Γ). The Γ conduction valley gets its own learnable cone term.
  - **Protocol:** select on a validation split (20% of each training pool); report held-out test MAE on the same 60-state test sets over the 5 splits of amendment (c).
  - Reported as exploratory, alongside the pre-registered model.
- **2026-10-25 09:26 PDT: full-HSE06 check of uniaxial ⟨110⟩ diamond** (author's request). Strict verify_hse settings, as used for agent states, at the ⟨110⟩ 14%, 16% and 18% uniaxial-stress states; the 2.0 eV crossing at full HSE06 is located by interpolation.
- **2026-10-25 12:12 PDT: exploratory idea 4, targeted diamond data** (author's request; recorded before running).
  - Candidates: 4,000 Latin-hypercube strains from the C benchmark distribution (|normal| ≤ 0.10, |shear| ≤ 0.08; new seed).
  - Selection: the 300 with the largest ensemble gap spread (minimum spacing 0.01) under the improved network (128×4, 1500 epochs) trained on the split-7 training pool only.
  - PBE DFT at the selected points. Each split's improved model is then retrained on its pool plus the new points and scored on the unchanged 5 test sets.
  - There is no random-points control, so a gain shows the effect of more data plus targeting together.
- **2026-10-25 19:46 PDT: diamond surrogate, option C** (author's request; exploratory).
  - **(A)** Test the step-3 best network (256×4, 3000 epochs) on the 5 diamond test splits (full data).
  - **(B) Representation fix:** predict absolute energies of 8 conduction valleys (Γ, L1–4, Δx–z) and 8 valence 'valleys' (Γ top, L1–4, X1–3) from the stored eigenvalues. Gap = min(CB) − max(VB). The Γ valence top and Γ conduction get learnable Bir–Pikus cone terms.
  - **(C)** B with the A architecture.
  - Same splits and test sets; reported as exploratory.
- **2026-10-25 20:58 PDT: option C, part B stopped.** The valence-plus-conduction representation was worse on both evaluated splits (137.6 vs 115.3, 117.5 vs 74.5 meV). Part C (B plus the A architecture) is therefore not run. Part A continues.
- **2026-10-26 01:05 PDT: random-data control for diamond** (author's request).
  - 300 Latin-hypercube strains from the C benchmark distribution (|normal| ≤ 0.10, |shear| ≤ 0.08), seed 20261026, which differs from the benchmark and the targeted-run seeds.
  - PBE DFT, then the improved network (128×4, 1500 epochs) retrained on each split's pool plus these points and scored on the unchanged 5 test sets.
  - Serves as the random control for the targeted-data run.
- **2026-10-26 08:53 PDT: 500 more random diamond points** (author's request).
  - Latin-hypercube strains from the C benchmark distribution, seed 20261027 (distinct from all earlier seeds).
  - PBE DFT, then the improved network (128×4, 1500 epochs) retrained on each split's pool plus all 800 random points (300 + 500) and scored on the unchanged 5 test sets.

- **2026-10-27: Elastic-stability reanalysis (post-campaign correction).** The `elastic_stability` tool used during all campaigns took finite differences of the Cauchy stress with respect to Green–Lagrange strain. After the campaigns, all 35 cached states with a full ±0.5% stencil were re-evaluated from the same DFT stresses with the second Piola–Kirchhoff stress (`eseagent/scripts/stability_recheck.py`, `runs/stability_recheck.json`). No acceptance verdict changed. Text and Fig. 6b use the reanalysis; verbatim transcript values (Fig. 3a) are from the original check. The criterion is described as a material-tangent screen, with phonon stability established separately by DFPT.
- **2026-10-27: Baseline name.** The "SCBO" arm implements constrained expected improvement (feasibility-weighted EI, random multistart), not trust-region SCBO; it is renamed cBO and cited to Gardner et al. (2014). No rerun.
- **2026-10-28 00:33 PDT: late cBO-naive run on H3** (author's request; after the data freeze, run after all other results were known). This partly reverses the 2026-10-24 15:20 scope cut, which had dropped H3 naive.
  - The naive formalization is the pre-registered rule (d) unchanged, exactly as encoded for H3 in scripts/scbo_arms.py since 2026-10-23: raw-PBE Eg ≤ 0, no stability constraint, full 6D strain (the e33 = 0 film constraint is not imposed), bounds ±0.10. Frozen cBO settings, 100 units, seed 0 (scripts/scbo_h3_naive.py; runs/scbo/H3_naive_s0.json).
  - Scored like the other H3 cBO runs: full HSE06 (variant A, valley-sampled) at the run's recorded best state (scripts/score_best_states.py). Because the H3 task fixes e33 = 0, a state with e33 ≠ 0 does not meet the H3 target whatever its gap (results.json scbo.*.subspace_ok). No elastic-stability check, as for the other H3 cBO states.
  - Outcome: best h = 1.009 meV/Å³ with PBE gap −0.124 eV, scissor-level gap 0.458 eV, full-HSE06 gap 0.414 eV and e33 = 0.023, so the target is not met (gap and subspace). References are unchanged.
- **2026-10-28 11:00 PDT: extended diamond learning curve** (exploratory; author's request; after the data freeze and after all other results were known).
  - Fig. 2c protocol unchanged (same five splits, 60 test states and training order; points up to 256 reproduce the original curve exactly). Training sets extended with 6,200 diamond states from the same Latin-hypercube box: C_rand (300), C_rand2 (500) and the overnight batches C_night1-18 (5,400); C_al excluded (active-learning selected). Training sizes 500, 1,000, 2,000, 4,000 and 6,456 (scripts/learning_curves_ext.py).
  - Outcome: valley-resolved diamond MAE 102.2 meV at 1,000 and 103.7 meV at 6,456 points (123.6 at 256); both baselines also level off. Fig. 2c shows up to 1,000 points (author's choice); the full curve is SI Table S4. No campaign result changes.
- **2026-10-28 22:15 PDT: three post-freeze checks** (author's request; after all other results were known; exploratory).
  - (a) **Scripted arm with the naive formalization**, seed 0 on H1-H3 (scripts/scripted_arm.py TASK 0 naive): same tools, 150-unit budget, schedule, verifier and one re-target as the expert scripted arm; formalization = pre-registered rule (d) (raw-PBE thresholds, full 6D strain, no stability in the search). Success = the pre-registered task criterion (full HSE06, elastic stability, e33 = 0 on H3). Outcome: H1 FAILED (full-HSE06 3.007 eV, then 1.933 eV after re-target), H2 VERIFIED after re-target (h = 0.36 meV/A^3, 0.74 eV), H3 FAILED (0.501 eV, e33 = -0.042). The expert-formalization scripted runs are unchanged (8 of 9).
  - (b) **Delta-line sampling check of full HSE06** for the H1 agent state and the <110> uniaxial states at 16% and 18% axial strain: HSE06 on a 12^3 grid (nq = 4, strict settings) gives the Delta correction at s = 2/3 and 5/6, bracketing the s = 0.75 point used throughout. Gaps change by at most 0.03 eV; the H1 agent state stays inside 2.0 +/- 0.03 eV (1.972-2.004 eV) (scripts/hse_delta_check.py).
  - (c) **Deterministic rule-based scoring** of the 22 formalization answers against the same key (scripts/rate_formalization_rules.py): mean 0.945; agreement with the Sonnet and Haiku judges 94% and 90%.
- **2026-10-30: extended Si learning curves** (author's request; after the data freeze and after all other results were known; exploratory).
  - Fig. 2a,b protocol unchanged (same five splits, 60 test states and training order; points up to the benchmark sizes, 247 and 259, reproduce the original curves exactly). Training sets extended with new random Si states from the same Latin-hypercube boxes and the same 10x10x10 k-mesh as the benchmark sets, new seeds: 770 small-strain and 760 wide-range points, one failed calculation each (scripts/si_extend.py). Training sizes 500, 1,000 and all states (1,016 and 1,018) (scripts/learning_curves_ext.py).
  - Outcome: valley-resolved MAE 8.8 meV (small strain) and 34.9 meV (wide range) at 1,000 points, vs 9.2 and 38.5 meV at the benchmark sizes; at 1,000 points both baselines (29.6/34.0 and 81.0/94.1 meV) remain less accurate than the valley-resolved model trained on 50 points. Fig. 2a,b now show up to 1,000 points like Fig. 2c; the values are SI Table S7. Table 3 (benchmark sizes) and all campaign results are unchanged.
