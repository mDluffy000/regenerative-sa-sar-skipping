# A. 55-nm Detector Schematic Characterization

## Paper-ready summary

The similarity detector combines two complementary-input regenerative sense-amplifier cores, a resistive threshold front-end, and CMOS decision/storage logic. The SA cores are adapted from Al-Qadasi et al., Fig. 1(b), whereas the complete dual-threshold similarity detector is this project's circuit combination, rather than a circuit reported in that paper. In nominal 55-nm CMOS schematic simulations at 1.2 V and 27 °C, the two resistor-ratio threshold settings are 19 and 38 mV, corresponding to the approximately 20- and 40-mV half-window classes. [S1–S3, S7]

Across 43 discrete common-mode settings spanning 0–1.2 V, all legal tested guard events were correctly classified: 390/390 for the 20-mV class and 342/342 for the 40-mV class, including ideal-source and 50-fF-per-input held-source fixtures. The common observation schedule provides a conservative decision-availability time of 760 ps, with direct output verification at 985 ps. The maximum accounted positive-delivered energy over a complete event is 57.10 and 61.31 fJ for the respective classes. These are nominal schematic results over the tested event set; the energy excludes analog input-source work and the internal losses of external timing generators. [S1, S4–S5]

## 1. Technology and simulation conditions

| Quantity | Verified value / interpretation | Source |
|---|---|---|
| CMOS technology | ICSprout 55-nm CMOS; MOS model include is the archived `icsprout_v1p15_usage_spe.lib` within the 55LLULP1233 PDK path | S1 |
| Supply / bias / reference | Both SA supply rails, logic supply, and VBIAS are 1.2 V; VSS = 0 V | S1 |
| Process corner | `tt_mos` (nominal TT) | S1 |
| Temperature | `temp=27`, `tnom=27` °C | S1 |
| Simulation level | Schematic; PDK MOS devices with ideal passive elements; no extracted parasitic netlist in this dataset | S1–S3 |
| Numerical settings | `scale=0.9`, `scalem=1`, `maxstep=1p`, `errpreset=conservative`; these are the archived settings, not an inferred physical gate length | S1 |
| Event period | 2.5 ns; slot 0 is initialization and excluded from event statistics | S4–S5 |
| Conservative decision-availability time | 760 ps from the beginning of the event; see Section 5 for the distinction from a sampling strobe | S4–S5 |
| Direct decision observation | 985 ps from event origin | S4 |
| Held-input capacitance | 50 fF on each held input, plus the detector's own 1-fF shunt on each input; the ideal-source fixture has no 50-fF holding capacitor | S1–S3 |
| Held-input isolation | Idealized switch opens at 35 ps and reconnects at 1900 ps; nominal transition is 1 ps, on conductance approximately 1 S and off conductance 1e−12 S | S4 |

Input definitions are

\[
V_{\mathrm{CM}}=(V_P+V_N)/2,\quad V_{\mathrm{DM}}=V_P-V_N,\quad
V_P=V_{\mathrm{CM}}+V_{\mathrm{DM}}/2,\quad V_N=V_{\mathrm{CM}}-V_{\mathrm{DM}}/2.
\]

The sweep coordinates refer to the applied input pair before holding. The actual held nodes can subsequently shift. [S4, S6]

## 2. Detector topology and attribution

**2 × complementary-input SA core + resistive threshold front-end + CMOS decision/storage logic.** The two threshold-offset branches feed `XP` and `XN`; `PN` and `NP` drive the CMOS NAND, whose `RAWF` output feeds the `WINDOW`-controlled D latch. `Q` is FAR and `QB` is NEAR. Both cores share CLK. [S1–S3]

Cite the regenerative core as `\cite{alqadasi2020rail}`: M. A. Al-Qadasi et al., “Rail-to-rail complementary input StrongARM comparator for low-power applications,” *IET Circuits, Devices & Systems*, vol. 14, no. 6, pp. 898–900, 2020, Fig. 1(b), DOI: 10.1049/iet-cds.2019.0361. The cited paper's technology, energy, speed and silicon results must not be assigned to this project's detector. Attribution of the core does not establish publication-priority novelty of the complete detector. [S7]

## 3. Similarity-window parameters

| Parameter | 20-mV class | 40-mV class | Source |
|---|---:|---:|---|
| Nominal target half-width | 20 mV | 40 mV | S1 configuration |
| Designed resistor-ratio half-width T | 19 mV | 38 mV | S1 configuration and S3 resistor network |
| Ri | 20 kΩ | 20 kΩ | S1 |
| Rb | 1263.157894736842 kΩ | 631.578947368421 kΩ | S1 |
| VBIAS | 1.2 V | 1.2 V | S1 |
| VSS | 0 V | 0 V | S1 |
| Tested NEAR guard differences | 0, ±16 mV | 0, ±36 mV | S5 |
| Tested FAR guard differences | ±24 mV | ±44 mV | S5 |

For the ideal unloaded resistor network,

\[
T=V_{\mathrm{BIAS}}\frac{R_i}{R_b},\quad
A_1-B_1=\frac{R_b}{R_i+R_b}(V_{\mathrm{DM}}-T),\quad
A_2-B_2=\frac{R_b}{R_i+R_b}(V_{\mathrm{DM}}+T).
\]

These equations follow from the archived front-end connections; the numerical T values are resistor-ratio design values, not extracted dynamic switching boundaries. The names “20 mV” and “40 mV” denote approximate **half-widths**, not the full width of the two-sided interval. [S1, S3]

The intended decision convention is

\[
|V_{\mathrm{DM}}|<T\Rightarrow\mathrm{NEAR},\qquad
|V_{\mathrm{DM}}|\ge T\Rightarrow\mathrm{FAR}.
\]

**Equality behavior and an exact measured transition threshold are `NOT AVAILABLE`.** The existing classifier excludes equality to its nominal target, and the tested guard points do not equal either the nominal target or the resistor-ratio T. All tested labels agree with both definitions away from the boundary. Consequently, the intended equality convention must not be presented as experimentally verified. [S5]

## 4. Common-mode coverage and functional validation

| Quantity | 20-mV class | 40-mV class | Source |
|---|---:|---:|---|
| Swept CM endpoints | 0–1.2 V | 0–1.2 V | S5 |
| Distinct CM points per fixture | 43 | 43 | S5 |
| Ideal-input legal events / passes | 195 / 195 | 171 / 171 | S5 |
| 50-fF held-input legal events / passes | 195 / 195 | 171 / 171 | S5 |
| Combined legal events / passes | 390 / 390 | 342 / 342 | S5 |
| Combined pass rate | 100% | 100% | S5 |
| CM points with all five signed guard cases | 37 | 31 | S5 |
| Endpoint range of full signed guard coverage | 0.012–1.188 V | 0.022–1.178 V | S5 |

The common grid, in volts, is: `0, .005, .010, .012, .015, .020, .022, .025, .030, .040, .050, .100, .150, .200, .250, .300, .350, .400, .450, .500, .550, .600, .650, .700, .750, .800, .850, .900, .950, 1.000, 1.050, 1.100, 1.150, 1.160, 1.170, 1.175, 1.178, 1.180, 1.185, 1.188, 1.190, 1.195, 1.200`. Full signed guard coverage means both NEAR-side and FAR-side samples around each boundary at the tested grid points; it is not an exact boundary extraction or a continuous-CM proof. [S5]

Legal input pairs satisfy

\[
0\le V_P,V_N\le1.2\ \mathrm V,\qquad
|V_{\mathrm{DM}}|\le2\min(V_{\mathrm{CM}},1.2-V_{\mathrm{CM}}).
\]

Thus, the allowed differential range contracts near either rail; at the two endpoints only zero difference is legal. Illegal requested guard pairs were omitted, not counted as passing events. [S5]

**Safe wording:** “At nominal TT conditions, all legal tested input events are correctly classified over the swept full common-mode range.” Accompany this sentence with the discrete grid and event counts. It does not establish arbitrary rail-to-rail differential operation, all possible legal input pairs, random histories, or behavior at untested boundaries. [S5]

Functional validation separately verifies the two core outputs, final classification, direct deadline observation and reset state. A valid output pair requires high > 0.96 V and low < 0.24 V. The historical aggregate `pass` field additionally checks an 80-fJ subtotal ceiling; all classification, capture, deadline and reset flags independently pass, so the functional claim does not depend on interpreting this energy budget as a circuit specification. [S4–S5]

## 5. Delay and timing

| Event-relative time | Archived action / evidence | Source |
|---|---|---|
| 20 ps | Input pair is updated, with 10-ps source transitions | S4 |
| 250 ps | CLK rising transition starts; transition duration 10 ps | S4 |
| 745 ps | Direct core-output capture (`CORECAP`); held-node observation `HCAP` | S4 |
| 750–760 ps | WINDOW falling transition closes the latch | S4 |
| 760 ps | Conservative service time, recomputed from last-invalid-output tracking and the end of latch closure | S4–S5 |
| 900 ps | CLK falling transition starts core reset | S4 |
| 985 ps | Direct Q/QB consistency observation (`DEADLINE`); held-node observation `HADC` | S4 |
| 1190 ps | Final classification (`RESULT`); validity tracking extends to just before this observation | S4 |
| 1200 ps | WINDOW rises to reopen the latch | S4 |
| 2490 ps | Reset state explicitly checked (`RESET`) | S4–S5 |
| 2500 ps | Complete event period and energy boundary | S4 |

Both classes and both input fixtures use this schedule. `stable_service_ps` is the larger of the relevant last-invalid time plus 0.25 ps and 760 ps; its maximum is 760 ps in every final group. Validity is monitored on a 0.25-ps grid over the archived evaluation interval. **There is no separate direct sampling strobe at 760 ps.** The actual direct output sample is at 985 ps, which itself satisfies the required decision observation below 1 ns. The service-time bound is not an optimized intrinsic SA regeneration delay. [S4–S5]

All tested resets pass by 2490 ps; the earliest reset-completion time is **`NOT AVAILABLE`**. A sub-nanosecond decision observation must not be converted into a 1-GHz sustained event-rate claim: the demonstrated event period is 2.5 ns. The 985-ps held-node sample is a timing-sensitivity observation, not a frozen system ADC sampling phase. [S4–S6]

## 6. Complete-event energy

All energies below are positive-delivered **accounted subtotals** integrated over the entire 2.5-ns event, including evaluation, storage, reset and continuous bias. Initialization slot 0 is excluded. The median is a descriptive median of the deterministic test set, not a workload-weighted average. [S4–S5]

| Class | Input fixture | Events | Minimum (fJ) | Test-set median (fJ) | Maximum (fJ) | Maximum including input-port work (fJ) |
|---|---|---:|---:|---:|---:|---:|
| 20 mV | Ideal source | 195 | 33.773951 | 50.489620 | 57.099733 | 57.874467 |
| 20 mV | 50 fF per input | 195 | 33.836080 | 50.443457 | 57.066360 | 107.504051 |
| 40 mV | Ideal source | 171 | 33.174210 | 50.380089 | 61.308847 | 63.747538 |
| 40 mV | 50 fF per input | 171 | 33.502439 | 50.409697 | 61.023172 | 114.107730 |

Source: S5, recomputed from the raw `ENERGY` records in S4. The two maximum columns need not occur in the same event.

| Subtotal-maximum record | CM (V) | VDM (mV) | Exact reported subtotal (fJ) |
|---|---:|---:|---:|
| `w_r2full_ideal_20_0`, slot 1 | 0 | 0.0 | 57.09973305114863 |
| `w_r2full_held_20_0`, slot 1 | 0 | 0.0 | 57.06636017920942 |
| `w_r2full_ideal_40_0`, slot 11 | 0.022 | -36.0 | 61.30884700830039 |
| `w_r2full_held_40_0`, slot 11 | 0.022 | -36.0 | 61.02317170475769 |

The headline therefore remains **57.1 / 61.3 fJ/event** when rounded to one decimal place. These are the per-class maximum subtotals across the ideal and held fixtures; the maxima in both cases occur in the ideal-source fixture. “Approximately 57–61 fJ/event” is acceptable only as coarse rounding with the precise values and accounting scope nearby; “≤61 fJ/event” is not supported. [S5]

The meter records net, positive-delivered and returned energy separately:

\[
E_{\mathrm{draw},k}=\int_{t_0}^{t_0+2.5\,\mathrm{ns}}\max(v_k i_k,0)\,dt,
\quad E_{\mathrm{net},k}=E_{\mathrm{draw},k}-E_{\mathrm{return},k}.
\]

The subtotal sums channels 0, 1, 2, 3, 6, 7 and 8: both SA supply rails, CMOS decision/storage logic supply, VBIAS rail, and the CLK, WINDOW and ACQ control ports. Control metering is upstream of the declared 100-Ω series fixture resistors. The final direct front-end does not use ACQ; its measured contribution is zero. The output loading declared by the fixture is reflected in source energy. Input-source channels 4 and 5 are excluded from the headline subtotal; adding them gives the separate last column above, including input preparation/hold recharge. Those totals are not automatically the detector's incremental system cost. [S1, S3–S5]

**Excluded / not determined:** the internal energy of external CLK/WINDOW/ACQ waveform generators, practical bias-generation losses, and the complete ADC/system implementation. In particular, external timing-generator internal energy is **`NOT AVAILABLE`**, rather than assumed zero or credited to an ADC. A workload-representative average incremental detector cost is **`NOT AVAILABLE`** in this characterization. [S1, S4, S8]

The source is the existing Verilog-A continuous-integral observers emitted into Spectre logs. Net = draw − returned was rechecked for every included channel/event. An independent completed PSF/Ocean energy re-integration is **`NOT AVAILABLE`**: the archived exporter logs report missing `CS`/`WS` waveform outputs and evaluation errors, and `FINAL_STATUS.json` records the crosscheck as incomplete. Existing PSF files remain archived remotely, but waveform presence is not an independent numerical energy validation. [S4, S8, S10]

## 7. Held-node disturbance

The matched comparison uses the same input sequence, 50-fF-per-input hold fixture, control schedule and observation times with and without the detector. The removed netlist deletes the front-end, both cores, NAND, latch and detector-output loads; its external input/hold fixture remains. The two 1-fF detector input shunts are removed with the front-end, so their contribution is part of this comparison. The representative detector netlists match the final-sweep DUT and fixture netlists apart from observer filename and run length. [S1, S3, S6]

For each held node,

\[
\Delta V_P(t)=V_{P,\mathrm{det}}(t)-V_{P,\mathrm{removed}}(t),\quad
\Delta V_N(t)=V_{N,\mathrm{det}}(t)-V_{N,\mathrm{removed}}(t),
\]
\[
\Delta V_{\mathrm{DIFF}}=\Delta V_P-\Delta V_N,\qquad
\Delta V_{\mathrm{CM}}=(\Delta V_P+\Delta V_N)/2.
\]

Both classes have 25 matched held events: five CM values, 0.025, 0.2, 0.6, 1.0 and 1.175 V, with the five class-specific signed guard inputs. This is a separate representative dataset, not disturbance characterization of all 732 functional events. The following maxima are maxima **across these events at a fixed observation time**, not peaks across a waveform. [S6]

| Class | Observation (ps) | Max single-ended absolute shift (mV) | Max absolute differential shift (mV) | Max absolute common-mode shift (mV) |
|---|---:|---:|---:|---:|
| 20 mV | 745 | 25.501211 | 5.810683 | 23.299018 |
| 20 mV | 985 | 19.693686 | 0.790145 | 19.693551 |
| 40 mV | 745 | 37.396137 | 6.971416 | 34.562979 |
| 40 mV | 985 | 34.882567 | 2.620189 | 34.882393 |

Source: S6, regenerated from connected-minus-removed `HCAP` (745 ps) and `HADC` (985 ps) log values.

| Class / time | Single-ended extremum witness | Differential extremum witness |
|---|---|---|
| 20 mV / 745 ps | `w_r2_held_20`, slot 22, CM=1.175 V, d=-24.0 mV: ΔVN=-25.501210983 mV | same job, slot 22, CM=1.175 V, d=-24.0 mV: ΔVDIFF=+5.810683483 mV |
| 20 mV / 985 ps | `w_r2_held_20`, slot 1, CM=0.025 V, d=0.0 mV: ΔVP=+19.693685825 mV | same job, slot 2, CM=0.025 V, d=-24.0 mV: ΔVDIFF=+0.790145197 mV |
| 40 mV / 745 ps | `w_r2_held_40`, slot 22, CM=1.175 V, d=-44.0 mV: ΔVN=-37.396137078 mV | same job, slot 22, CM=1.175 V, d=-44.0 mV: ΔVDIFF=+6.971416425 mV |
| 40 mV / 985 ps | `w_r2_held_40`, slot 1, CM=0.025 V, d=0.0 mV: ΔVP=+34.882567086 mV | same job, slot 2, CM=0.025 V, d=-44.0 mV: ΔVDIFF=+2.620188876 mV |

The disturbance is a **signed shift**, not a universally positive noise magnitude. At 985 ps, the combined P/N signed range is −16.799788 to +19.693686 mV for the 20-mV class and −32.369175 to +34.882567 mV for the 40-mV class. The common-mode component is substantial, while the maximum differential residual is smaller at 985 ps than at 745 ps. These observations do not imply that every event has recovered, or that common-mode errors are harmless to an eventual ADC. [S6]

The CSV's primary disturbance fields use the 985-ps maximum absolute shifts. **Evaluation-wide temporal peak, shift at exactly 760 ps, reset/recovery temporal peak and next-event residual maxima are `NOT AVAILABLE` in the audited fixed-time disturbance table.** The present data are deterministic input/history-dependent nominal simulations; mismatch distributions, transient-noise distributions and manufacturing quantiles are `NOT AVAILABLE` for this final two-setting characterization. No noise, PVT or Monte Carlo experiment was added. [S6, S8]

## 8. Circuit size and complexity

| Quantity per detector | Both classes | Source / boundary |
|---|---:|---|
| MOS transistors | 60 | S2 and S9 |
| MOS per SA instance | 19, including its output buffers | S2; two instances contribute 38 |
| CMOS NAND + D latch | 4 + 18 = 22 MOS | S2 |
| Ideal resistors | 8: four Ri and four Rb | S3 and S9 |
| Ideal capacitors | 2: one 1-fF input-to-VSS shunt per input | S3 and S9 |
| Physical layout area | `NOT AVAILABLE` | No verified physical area for this frozen schematic pair |

The count is the detector, excluding the testbench's three 100-Ω control-series resistors, six 2-fF output loads, external ideal sources/switches, and the two 50-fF hold capacitors in the held fixture. The OA readback independently contains 31 PMOS, 12 low-Vt NMOS and 17 standard-Vt NMOS, plus the eight `analogLib/res` and two `analogLib/cap` instances in each of `CISA2020_WIDE20` and `CISA2020_WIDE40`. The resistor and capacitor implementations are ideal passive devices, not verified physical resistor/capacitor layouts. [S1–S3, S9]

## 9. Scope disclosure

These are nominal TT **schematic-only** results for the frozen detector pair, with ideal passive devices and an idealized hold fixture. No new PEX, silicon measurement, process-variation or transient-noise characterization is included. Internal over-voltage is still recorded: final-sweep log summaries report A2 up to 1.464 V in the 20-mV ideal fixture and B1 up to 1.463 V in the 40-mV ideal fixture; duration and device-terminal reliability qualification are **`NOT AVAILABLE`**. External timing-generator internal energy is excluded. These limitations are disclosures, not additional work items. [S1–S4, S8, S11]

## Source ledger and audit record

All local source links below are read-only inputs to this summary. Event provenance is keyed by `(job, slot)`; each key maps directly to matching tagged records in that job's `spectre.out`. Numerical statistics were recalculated from those records, without launching a simulator.

**S1 — Frozen sweep conditions and parameter settings:** [w_r2full_held_20_0.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_0.scs), [w_r2full_held_40_0.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_40_0.scs), [w_r2full_ideal_20_0.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_20_0.scs), [w_r2full_ideal_40_0.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_40_0.scs), [w_r2full_held_20_0_config.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_0_config.json), [w_r2full_held_40_0_config.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_40_0_config.json).

**S2 — MOS hierarchy and decision/storage connections:** [cisa_b.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/cisa_b.scs), [detector_logic.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/detector_logic.scs).

**S3 — Resistive front-end and ideal input shunts:** [window_front_r2.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/window_front_r2.scs).

**S4 — Timing/meter observers and raw final-sweep records:** [w_r2full_held_20_0.va](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_0.va), [jobs_wr2full.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/jobs_wr2full.json).

**S5 — Functional and energy tables, event membership and historical extraction rules:** [final_events.csv](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/final_events.csv), [final_summary.csv](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/final_summary.csv), [final_by_cm.csv](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/final_by_cm.csv), [window_analyze.py](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/window_analyze.py), [w_r2_validate.py](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_validate.py).

**S6 — Matched held-node comparison:** [held_disturbance_vs_removed.csv](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/held_disturbance_vs_removed.csv), [held_analyze.py](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/held_analyze.py), [w_r2_held_20.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_20.scs), [w_r2_removed_20.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_removed_20.scs), [w_r2_held_40.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_40.scs), [w_r2_removed_40.scs](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_removed_40.scs), [w_r2_held_20_events.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_20_events.json), [w_r2_held_40_events.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_40_events.json), [w_r2_held_20.va](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_20.va), [w_r2_held_40.va](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_40.va).

**S8 — Frozen status and prior energy crosscheck limitations:** [FINAL_STATUS.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/FINAL_STATUS.json), [REPORT.md](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/REPORT.md), [read_integrals.ocn.log](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/read_integrals.ocn.log), [read_quick.ocn.log](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/read_quick.ocn.log).

**S9 — Independent schematic OA instance readback:** [OA_VERIFIED_WIDE.json](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/OA_VERIFIED_WIDE.json).

**S11 — Observed internal over-voltage, simulation-wide summary (not event-resolved reliability analysis):** [w_r2full_ideal_20_2/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_20_2/spectre.out), [w_r2full_ideal_40_1/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_40_1/spectre.out).

**S7 — Core reference:** [local paper](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/reference/paper.pdf), p. 898, Fig. 1(b) and bibliographic header.

**S4 raw-log inventory:**

| Job | Included event slots | Raw log |
|---|---|---|
| `w_r2full_ideal_20_0` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_20_0/spectre.out) |
| `w_r2full_ideal_20_1` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_20_1/spectre.out) |
| `w_r2full_ideal_20_2` | 1–35 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_20_2/spectre.out) |
| `w_r2full_ideal_40_0` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_40_0/spectre.out) |
| `w_r2full_ideal_40_1` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_40_1/spectre.out) |
| `w_r2full_ideal_40_2` | 1–11 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_ideal_40_2/spectre.out) |
| `w_r2full_held_20_0` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_0/spectre.out) |
| `w_r2full_held_20_1` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_1/spectre.out) |
| `w_r2full_held_20_2` | 1–35 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_20_2/spectre.out) |
| `w_r2full_held_40_0` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_40_0/spectre.out) |
| `w_r2full_held_40_1` | 1–80 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_40_1/spectre.out) |
| `w_r2full_held_40_2` | 1–11 | [spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2full_held_40_2/spectre.out) |

**S6 matched raw logs:** [w_r2_held_20/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_20/spectre.out), [w_r2_removed_20/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_removed_20/spectre.out), [w_r2_held_40/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_held_40/spectre.out), [w_r2_removed_40/spectre.out](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/cisa2020_b_20260910/cm_opt/w_r2_removed_40/spectre.out).

**S10 — Existing remote PSF archive (read-only inventory verified during this audit):** root `/home/liuyufei/cisa2020_b_20260910/cm_opt/` on `server25-wan`. Every final job listed in S4 retains `psf/tran.tran.tran`, `psf/tran.tran.tran.psfxl`, `psf/tran.tran.tran.sig` and `psf/logFile`; the connected/removed representative jobs retain their PSFXL data as well. The representative `w_r2full_held_20_0/psf/logFile` identifies Spectre `25.1.0.313.isr7`, APS, transient analysis and a 202.501-ns total run. This duration contains multiple events and is not the event period. Remote hashes of `w_r2full_held_20_0.scs`, its `spectre.out`, and the held/removed 20-mV representative logs match their local copies. Remote copies of the two final derived CSVs were not present at this root; this audit uses the local CSVs and validates them against archived raw logs. This audit verified PSF availability/metadata, not a new full waveform re-integration.

**Audit outcome:** all 732 final event records, all four functional/energy summary groups and all 50 matched disturbance records agree with their raw-log reconstructions. Both threshold classes use the same schedule and nominal conditions. The representative and full-sweep DUT/fixture netlists agree except for observer source filename and run duration. No source was modified and no new simulation was run.

**Local source SHA-256 fingerprints:**

| File | SHA-256 |
|---|---|
| `final_events.csv` | `96e8af31879aad8f765258c51469e8ba7a2f49a0926bec47cbaafc8b7750c823` |
| `final_summary.csv` | `344370253c1aa053d91b54e6495de6cacc2f3a2857608d070a6de56b467de496` |
| `held_disturbance_vs_removed.csv` | `d5fb48da6aecd1dd33ac6c62755ec324f33a25a869fb4afbcd754d945fa8220b` |
| `jobs_wr2full.json` | `558685a9595b0c26adb94f3eda831a334b5c6c3f5b3ba5af783fe665c5eb9ca6` |
| `cisa_b.scs` | `72a1e2e627ee7148578fc20b22aa1d50a9824af66403c98b1e10895eb36573c5` |
| `detector_logic.scs` | `45a9f01afd41704d042115bf9e259281f54b8fbe43d9e31663a3543c380208e0` |
| `window_front_r2.scs` | `917024dd0b035c49d06dd165cc9fd9ac4fb8e06eeaa3eaf25096846f831f072c` |
| `OA_VERIFIED_WIDE.json` | `5aac7d33147a0070c15e2050a3dd253641f7852b980fcfdd1113b6b2487b5e28` |

## Companion CSV interpretation

`window_mV` is the nominal 20/40-mV class; `actual_T_mV` is the 19/38-mV resistor-ratio half-width. `temp` is in °C and voltages without a suffix are in volts. `legal_events` combines ideal and held fixtures, whereas `CM_points` counts distinct common-mode settings, not repeated fixtures. `pass_rate=1.0` means every included event passed. `decision_ps=760` denotes the conservative service bound; `direct_observation_ps=985` is the sampled output check. `energy_fJ` is the per-class maximum accounted positive-delivered complete-event subtotal. The two main disturbance columns are maxima of absolute signed matched differences at `disturbance_time_ps=985`, over the separate 25-event held dataset. They are not full-sweep or waveform-peak bounds. Exact floating-point values are retained for traceability, not as claims of measurement precision.
