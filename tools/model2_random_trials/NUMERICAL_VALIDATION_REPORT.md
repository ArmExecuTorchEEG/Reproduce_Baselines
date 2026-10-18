# Model 2 numerical validation investigation

Investigation on `feature/model2-random-eeg-trials`, 2026-10-17 (PDT).
**No model, trained weights, quantization parameters, production streaming code,
frontend design, original firmware, physical board or serial connection was changed.**
Only investigation tools/reports and a separate ignored numerical FVP project were
created. No commit/push or physical deployment was performed.

## Conclusion

The 40 discrepancies are explained by **average-pooling rounding semantics in the
float Q/DQ reference versus the integer TOSA/Ethos-U85 computation**, not a streaming
input/alignment defect. This conclusion has direct operator-level evidence:

1. Every one of the288 FVP output levels equals the independently evaluated, ORIGINAL
   saved TOSA integer graph output. Zero raw integer mismatches over all 144 trials.
2. All 9 quantized weight/bias arrays match between the saved Q/DQ and TOSA graph,
   after the explicit backend tensor layout transformations.
3. Tracing a passing and a failing trial shows both pre-pooling convolutions and
   their requantized activations are identical. The first differences appear at
   the first AVG_POOL2D followed by Q/DQ requantization.
4. A diagnostic interpreter applying only integer pooling rounding at the two
   existing pool outputs makes the saved Q/DQ graph agree with TOSA/FVP on all 144
   final raw outputs. No convolution, weights, biases or scales were changed.
5. A fresh separate FVP run of passing/failing/tied cases in forward/reverse order
   reproduced identical outputs and exact TOSA agreement, excluding order/history
   as the cause in these representative tests.

This is a confirmed explanation for this tested fixed model and eligible pool.
It is not a universal bound for future models, preprocessing changes or new inputs.

## Output quantization and raw values

The actual serialized PTE execution plan has an EthosU delegate returning a signed
INT8 `(1,2)` tensor, followed by CPU `quantized_decomposed::dequantize_per_tensor`:

| Parameter | Input tensor | Output logits |
|---|---|---|
| Scale |0.04422667250037193|**0.052609603852033615**|
| Zero point |−10|**−5**|
| Integer range |−128…127|−128…127|
| Shape |(1,1,22,500)|(1,2)|

Output formula: `logit = float32(q - (-5)) * float32(0.052609603852033615)`.
The final Q/DQ quantizer and serialized PTE have exactly the same output scale and
zero point. Raw desktop values were captured directly from its final quantizer;
TOSA outputs were directly returned as INT8. The existing FVP wire protocol returns
float32 logits, not its intermediate raw buffer. FVP raw levels were therefore
uniquely reconstructed and validated by **bit-exact float32 dequantization for all
288 values**. Do not describe them as directly byte-captured FVP raw tensors.

| Count | Zero integer steps | One integer step | More than one |
|---|---:|---:|---:|
| Trials (maximum of two logits) |104|40|0|
| Individual logits |242|46|0|

Signed `FVP−Q/DQ` raw differences:242 zero,27 plus-one,19 minus-one.
34 trials differ in one logit;6 differ in both. All raw differences are exactly one
integer level. Dequantized subtraction differs from the ideal step slightly due
to float32 representation: maximum0.05260968208312988, about7.82e-8above the scale.
There is no extra multi-step output error, off-grid float value or incorrect zero
point in this corpus.

## Direct pooling evidence

The original saved desktop quantizer implementation uses
`torch.round(input * (1.0 / scale)) + zero_point`. `torch.round` uses ties-to-even.
For failing trial `S1/1test/run0/cue6623` (dataset index3), at first-pool output
position NHWC `[0,0,0,7]`:

- Four raw inputs sum to−510, with activation zero point−128.
- Centered sum is2; average is0.5quantized activation units.
- Desktopfloat pool is0.01062646321952343; divided/scaled for quantization it is
  exactly0.5. Ties-to-even rounds to0; adding−128 yields **−128**.
- Integer TOSA pooling rounds that half-way value upward to1; adding−128 yields
  **−127**. Its raw-average equivalent is `floor(sum/4 + 0.5)` here.

This behavior follows TOSA1.0.0§2.3.2,pp28–30: integer AVG_POOL2D uses
`reciprocal_scale(count)` with single-round `apply_scale_32`. §4.5.5,pp149–150 defines
its positive rounding offset and arithmetic shift. For these kernel4/8pools,
rounding exact half-way values differs from ties-to-even. Both pool activation
zero points are−128. Intermediate quantized differences propagate through later
layers; a one-activation-step difference is not universally limited to one final
output step. Here the independently executed full TOSA graph establishes the
correct final values for all 144 trials.

Sources: [TOSA1.0.0 specification](https://www.mlplatform.org/assets/pdf/tosa_spec_1_0_0.pdf),
CMSIS document `web/www-mlplatform-org/2b93c5c0a184`,§2.3.2pp28–30 and§4.5.5pp149–150;
installed PyTorch `torch/ao/quantization/fx/_decomposed.py:49–85`;
installed ExecuTorch `backends/arm/operators/op_tosa_rescale.py`.

The diagnostic interpreter is recorded in `pool-rounding-counterfactual.json`:
**144/144 final integer outputs exact**,0remaining TOSA differences after changing
only the two pooling output rounding rules IN MEMORY. This is not a modified model
or a replacement production inference pipeline.

## Integrity, source identity and ordering checks

- Original selected checkpoint SHA256:
  `63e3479baa0b00cadce5ee7906f25eee873ba561f9040cd07505f5d32b7a51d9`.
  Reload produces the saved29 development float logits bit-exactly,1,938 parameters.
- PTE SHA256 remains
  `644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298`.
  Extracted C-array model bytes in original, streaming and numerical projects
  all equal that saved13,072-byte PTE.
- All 5 weight arrays and4 bias arrays equal TOSA constants in their corresponding
  NHWC/depthwise/linear layouts. No model regeneration or quantization occurred.
- Normalized Session2float32windows are bit-exact to original saved test tensors;
  input INT8quantization recomputes exactly; browser input asset bytes agree.
- Training-only mean/std, SOS coefficients, 250 Hz cue-relative[1,3)window and all
 22 channel names/order agree with original preprocessing implementation and params.
- Every Session2frame is checked by session/request/trial index, full sourceID,
  length, wire CRC, payload CRC and trial SHA256. References map by source identity;
  no alignment is inferred from arrival order, class or ground truth.
- Original 144 desktop Q/DQ logits reproduce bit-exactly from the saved graph.
- All 136 original rollback-source files still match initial checksums. The validated
  demo branch remains `c85ab70f6030243283431f3a12b1a7ea87aa8b72`.

## Determinism, classes and decision margins

The existing 151-case FVP run (original 4 + held-out 144 +3 repeats) has exact repeated
results. A fresh isolated FVP test uses indices
`[0,3,2,5,32,64,64,32,5,2,3,0]`: passing0, both-logit failing3, second-logit failing2,
first-logit failing5/32including opposite signed deltas, and tied64. All 12 results
match TOSA exactly; repeated/reversed cases are identical. No input selection was
based on classification correctness.

**Predicted classes changed:0/144.** Agreement with quantized desktop is144/144;
ground-truth accuracy remains114/144=79.1667%. These are separate metrics.

Signed margin is `logit1−logit0`. Relative to Q/DQ, integer margin changes are:
104 unchanged,26 minus-one-step,6 minus-two-steps,8 plus-one-step. Thus40 margins change,
but none crosses the class decision boundary. The largest change is two output
steps (approximately0.10521921before float32 subtraction rounding).

Trial64, `S1/1test/run2/cue71202`, has equal logits0.21043841540813446in both
implementations. Both use the existing tie-to-class0rule. It is NOT evidence of
a robust margin. Smallest nonzero margin is one output step. No noise robustness,
clinical performance or generalization claim follows from these comparisons.

## Approved numerical criteria

Do not increase the saved Q/DQ threshold merely to make it pass.

1. **Primary backend correctness:** require exact raw INT8equality against the
   independently evaluated saved TOSA integer graph. This is zero-step tolerance,
   and passes288/288 values here. For physical E8 float responses, verify finite
   outputs, correct output quantization grid, IDs/checksums and the existing0.01
   absolute logit check against dequantized TOSA/FVP values. Since one output step
   exceeds0.01, this continues to reject a one-step backend disagreement.
2. **Classification:** independently require144/144 Q/DQ class agreement for this
   fixed pool; continue reporting actual truth accuracy separately. Do not use
   correct-class selection or invent a response on timeout.
3. **Retained cross-semantics diagnostic:** the ORIGINAL0.01 Q/DQcomparison remains
   **FAIL on40/144**. Report normalized step differences alongside it. One output
   step is the observed bound for this enumerated pool, NOT a universal theoretical
   tolerance for propagated quantization error, and is not silently adopted as a
   new pass threshold.

Under these user-approved backend-aware criteria, the implementation is numerically
acceptable for a **controlled initial physical test**, subject to user approval.
No numerical firmware/model correction is indicated. If a desktop reference is
needed for compiled behavior, use the saved integer TOSA graph; do not change the
trained model, firmware arithmetic, existing Q/DQreference or displayed logits to
force agreement.

The old deployment manifest remains unchanged as historical evidence. The explicitly approved policy is recorded separately in `validation/approved-validation-policy.json`; no physical action is authorized.

## Remaining risks and preservation

FVP does not establish physical UART4IRQ/FIFO margins, USB scheduling, cache/memory
behavior on E8, sustained runtime allocation, debugger reliability or hardware
logit agreement. Raw FVP intermediate bytes were not directly logged (see exact
reconstruction evidence above). TOSA tools2026.5.0read the saved1.0graph using a
backward-compatible1.1.0dserializer and report TOSA_VALID; the version warning is
recorded. A backend conformance pass for this model is not a universal error bound.

Physical testing should use the existing unchanged streaming image, a small fixed
sequence including these representatives, Chrome as sole UARTowner at115200/8N1,
and the independently saved TOSA/FVP references. Exact application load/start
scope and rollback procedure remain in `firmware/streaming/README.md`. Loading is
a physical modification and requires separate explicit authorization; do not erase,
provision, change SE/debug stubs or automatically retry/reset on a failure.

Current DevKitHEX SHA256 remains
`84ca339645e211bffdeed17d60d6f2fee7e448829b3e44506e4da81394a13afa`;
application vector/load base0x80200000. The original four-trial firmware archive,
validated frontend branch and working streaming implementation remain unchanged.

## Evidence and reproduction

Local ignored evidence: `artifacts/model2-numerical-investigation/`:
`summary.json`, `all-trials.json`, `failing-trials.csv`, `integer-references.npz`,
`backend-reference.npz`, `operator-trace.json`, `pool-rounding-counterfactual.json`,
`embedded-model-verification.json`, `checkpoint-reload.json`,
`proposed-validation-criteria.json`, plus intermediate tensor traces.
Fresh FVP evidence: `artifacts/model2-numerical-fvp/{expected.json,comparison.json,project/fvp-stream.log}`.
Original151run/QDQdiagnostic remains in `artifacts/streaming-firmware/` untouched.

New tools in `tools/model2_random_trials/`:
`investigate_numerics.py`, `trace_numerics.py`, `pool_counterfactual.py`,
`prepare_numerical_fvp.py`, `compare_numerical_fvp.py`.
Use the existing official Arm Python environment with`-B`; no dependency installed.
Run the first3tools with `--baseline-repo /path/to/original --out /NEW/output`;
trace/pool tools use the completed audit output. Audit refuses an existing output
directory; trace refuses existing per-trial directories. Numerical FVP preparation
uses `--out artifacts/model2-numerical-fvp/project`and refuses overwriting it.
Build/run FVP only through CMSIS Developer Assistant, explicit targetSSE-320-U85
and frontend window92823. Comparator uses `--directory artifacts/model2-numerical-fvp`.
Do not invoke any physical load/run/reset action during reproduction.

## All 40 trials outside the unchanged 0.01 diagnostic

Delta columns are exact signed raw levels (`FVP−desktop`); multiply by the output
scale for idealized logit differences. Full float32 values and actual signed
subtraction per logit are in `artifacts/model2-numerical-investigation/failing-trials.csv`.

| Dataset index | Source trial ID | Δ logit0 (levels) | Δ logit1 (levels) |
|---:|---|---:|---:|
|2|`S1/1test/run0/cue4671`|+0|-1|
|3|`S1/1test/run0/cue6623`|+1|-1|
|5|`S1/1test/run0/cue10742`|+1|+0|
|14|`S1/1test/run0/cue46711`|+1|+0|
|18|`S1/1test/run0/cue65004`|+1|-1|
|21|`S1/1test/run0/cue71202`|+1|+0|
|29|`S1/1test/run1/cue20544`|+0|-1|
|32|`S1/1test/run1/cue28322`|-1|+0|
|36|`S1/1test/run1/cue38618`|+0|-1|
|37|`S1/1test/run1/cue40688`|+1|+0|
|39|`S1/1test/run1/cue54733`|+0|+1|
|41|`S1/1test/run1/cue62912`|+1|+0|
|43|`S1/1test/run1/cue77157`|+1|+0|
|52|`S1/1test/run2/cue12659`|+1|+0|
|54|`S1/1test/run2/cue20544`|+0|-1|
|55|`S1/1test/run2/cue26355`|+1|-1|
|57|`S1/1test/run2/cue40688`|+0|-1|
|62|`S1/1test/run2/cue56708`|+1|+0|
|66|`S1/1test/run2/cue75194`|+0|-1|
|74|`S1/1test/run3/cue12659`|+1|+0|
|79|`S1/1test/run3/cue32450`|+0|+1|
|81|`S1/1test/run3/cue42768`|+0|+1|
|86|`S1/1test/run3/cue60805`|+1|-1|
|87|`S1/1test/run3/cue62912`|+0|-1|
|89|`S1/1test/run3/cue69138`|+1|+0|
|91|`S1/1test/run3/cue75194`|+1|-1|
|92|`S1/1test/run3/cue79082`|+1|+0|
|95|`S1/1test/run3/cue95257`|+1|+0|
|101|`S1/1test/run4/cue20544`|+1|-1|
|102|`S1/1test/run4/cue22439`|+1|+0|
|107|`S1/1test/run4/cue48664`|+0|+1|
|111|`S1/1test/run4/cue60805`|+1|+0|
|113|`S1/1test/run4/cue69138`|+0|-1|
|115|`S1/1test/run4/cue77157`|+1|+0|
|118|`S1/1test/run4/cue89156`|+1|+0|
|120|`S1/1test/run5/cue2753`|-1|+0|
|125|`S1/1test/run5/cue22439`|+0|-1|
|126|`S1/1test/run5/cue24411`|-1|+0|
|138|`S1/1test/run5/cue67095`|+1|+0|
|142|`S1/1test/run5/cue81163`|-1|+0|

## Approval record

The user explicitly approved the three numerical criteria above. Physical deployment remains unauthorized. Compact per-trial outputs, operator evidence and the approved policy are preserved in `validation/`; original ignored diagnostic results remain unchanged.
