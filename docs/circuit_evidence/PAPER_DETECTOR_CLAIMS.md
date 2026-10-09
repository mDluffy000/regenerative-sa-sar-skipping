# Detector claims for Section A

Evidence IDs S1–S11 refer to the linked source ledger in [PAPER_DETECTOR_CHARACTERIZATION.md](/Users/liuyufei/.codex/.chatgpt-projects/g-p-6a671d492b18819189c8bb31c8a6025d/paper_detector_characterization/PAPER_DETECTOR_CHARACTERIZATION.md). The accompanying CSV preserves exact source-derived values.

## SAFE TO CLAIM

- **55-nm schematic detector:** nominal `tt_mos`, 1.2-V supplies/bias and 27 °C; two complementary-input SA cores plus a resistive threshold front-end and CMOS NAND/latch logic. These are schematic simulations of this project's full detector combination. [S1–S3]
- **Core attribution:** the SA core is adapted from Al-Qadasi et al. (2020), Fig. 1(b); cite `\cite{alqadasi2020rail}`. The complete similarity detector is this project's combination, not an existing full detector attributed to that paper. [S2, S7]
- **19/38-mV designed half-windows:** Ri = 20 kΩ, Rb = 1263.157894736842 / 631.578947368421 kΩ, VBIAS = 1.2 V, with T = VBIAS Ri/Rb. Call these the approximately 20/40-mV threshold classes. They are ideal resistor-ratio targets rather than extracted exact switching thresholds. [S1, S3]
- **Full tested CM coverage:** “At nominal TT conditions, all legal tested input events are correctly classified over the swept full common-mode range.” State the 43 discrete CM points spanning 0–1.2 V and the 390/390 and 342/342 passes, including 195/195 and 171/171 events in each input fixture. Full signed near/far guard tests are supported at tested CM points spanning 0.012–1.188 V and 0.022–1.178 V, respectively. [S4–S5]
- **Decision observation below 1 ns:** the common direct observation at 985 ps passes all final events. The archived stability observer supports a conservative service bound of 760 ps; reset is verified by 2490 ps in a 2500-ps event period. [S4–S5]
- **Maximum accounted complete-event energy: 57.1 / 61.3 fJ/event.** These are rounded per-class maxima over the ideal/held test set, integrated over the complete 2.5-ns event. Held-fixture maxima are 57.066360 / 61.023172 fJ. Both SA rails, CMOS logic, bias and measured control-port work are included; input-source work and external timing-generator internal losses are excluded. [S1, S4–S5]
- **Fixed-time matched held-node shifts:** at 985 ps, maximum absolute single-ended shifts are 19.693686 / 34.882567 mV and maximum absolute differential shifts are 0.790145 / 2.620189 mV, each over 25 matched 50-fF-per-input events. At 745 ps, the corresponding single-ended maxima are 25.501211 / 37.396137 mV and differential maxima are 5.810683 / 6.971416 mV. The underlying shifts are signed and can be positive or negative. [S6]
- **Size:** 60 MOS, eight ideal resistors and two ideal 1-fF capacitors per detector, excluding the external testbench. Physical layout area is `NOT AVAILABLE`. [S1–S3, S9]

### Headline adjudication

**Supported with explicit scope:**

> A 55-nm CMOS schematic similarity detector with 19/38-mV resistor-ratio half-windows correctly classifies all legal tested guard events across 43 common-mode settings spanning 0–1.2 V at nominal TT conditions. Maximum accounted complete-event energy is 57.1/61.3 fJ for the two settings, excluding analog input-source work and external timing-generator internal losses. [S1–S5]

“55-nm schematic detector, full tested CM coverage, 19/38-mV windows, approximately 57–61 fJ/event” is a coarse shorthand for this qualified statement. Prefer the precise 57.1/61.3 figures in the paper; do not turn rounding into a 61-fJ upper bound.

## DO NOT CLAIM

- Arbitrary rail-to-rail differential operation; correctness for every possible legal input pair; continuous-CM coverage; all input histories; or process-independent rail-to-rail performance. Near the rails, the tested differential set contracts to maintain legal input voltages; only d = 0 is legal at CM = 0 or 1.2 V. [S5]
- Experimentally verified classification at exactly |VDM| = T, a precisely extracted 19/38-mV switching boundary, or calibrated offset/hysteresis. Those results are `NOT AVAILABLE`; only the specified guard points were tested. The intended “equality → FAR” convention is not verified. [S3–S5]
- A direct output sampling strobe at 760 ps, a minimum/optimized intrinsic SA delay, a 1-GHz sustained event rate, or a reset-completion time earlier than the recorded check. The demonstrated direct sample is 985 ps and the event period is 2.5 ns. [S4–S5]
- 57.1/61.3 fJ as measured silicon energy, PEX energy, net energy, workload-average energy, or total end-to-end system overhead. Nor may external clocks/periphery be assumed free because an ADC might supply them. Full external timing-generator internal energy and workload-representative average incremental cost are `NOT AVAILABLE`. [S1, S4–S5, S8]
- A completed independent PSF/Ocean energy crosscheck. It is `NOT AVAILABLE`; the archived export attempts failed. PSF file presence does not change this status. [S8, S10]
- The fixed-time disturbance maxima as evaluation-wide peaks, full-CM-sweep worst cases, universally nonnegative noise, reliable manufacturing tails, mismatch distributions or per-event transient-noise probabilities. Waveform-peak and variation/noise characterizations for this audited final pair are `NOT AVAILABLE`. [S6, S8]
- Zero disturbance, complete held-node recovery by the output sample, or a frozen actual ADC sampling phase at 985 ps. Significant signed common-mode residual remains in the matched dataset. [S6]
- A verified layout area, physical passive implementation, post-layout validation or silicon result for this frozen pair. Real layout area is `NOT AVAILABLE`; the characterized passives are ideal. [S1–S3, S8–S9]
- Absence of internal over-voltage or demonstrated oxide/reliability safety. Final raw logs still show internal/front-end excursions above the 1.2-V supply; stress duration and reliability qualification are `NOT AVAILABLE`. [S11]
- That Al-Qadasi et al. already published this complete similarity detector, or that their original technology/energy/speed/silicon results validate this project's implementation. Attribute only the reused core topology and distinguish the project-specific combination. [S2, S7]

No additional simulations or detector optimization were performed for these files. The characterization task ends with this evidence summary.
