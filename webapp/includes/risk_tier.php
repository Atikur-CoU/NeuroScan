<?php

/**
 * Confidence-tiered risk classification shown alongside the AI prediction.
 *
 * Hemorrhagic and ischemic are NOT tiered symmetrically -- see
 * GRAD_CAM_HEATMAP.md (project root) for the full research and sourcing
 * behind this. In short: hemorrhagic stroke must always be ruled out
 * before thrombolytic (tPA) therapy because giving tPA to a hemorrhagic
 * patient actively worsens the bleed, whereas withholding it from an
 * ischemic patient only costs time -- so a missed/under-triaged
 * hemorrhagic call is a categorically worse failure than the reverse.
 * Real FDA-cleared hemorrhage-detection AI (Aidoc, Viz ICH, Rapid ICH) is
 * deliberately tuned for high sensitivity over specificity for exactly
 * this reason, tolerating substantial false-positive rates rather than
 * risk missing a bleed. Accordingly: a winning hemorrhagic call is NEVER
 * downgraded to "uncertain" (mathematically it's already >33.3% just by
 * winning the 3-way softmax) and only needs 55% confidence to reach
 * "high risk", vs. ischemic's unchanged 90%/70% bar -- ischemic is also
 * the model's weaker class (final model R30A s123 on the pooled held-out
 * test: ischemic AUROC 0.963 vs. hemorrhagic 0.990), so a given confidence
 * number is less trustworthy there.
 *
 * These thresholds are design choices, not optimized or validated against
 * outcomes. With the deployed rule (4-view TTA, plain argmax, uncalibrated
 * probabilities) the paper reports them on the final model (post-hoc
 * analysis B4, pooled held-out test): 92.0% of hemorrhage slices alerted,
 * 80.6% of ischemic slices alerted, false alerts on 9.3% of normal slices,
 * 17.5% of slices in the uncertain tier. Lesions predicted as normal get no
 * alert at all, whatever the tier.
 *
 * $likelyNotAScan (nullable -- see migrate_004_ood_guard.sql): the model
 * has no "none of the above" option and will confidently force any image
 * into one of the three classes, so a wrong-file upload (a photo, a
 * screenshot) can otherwise reach a "High risk" badge with an audio
 * alert despite being completely unrelated to a brain scan. When true,
 * short-circuits to a distinct "unreliable" tier regardless of the
 * computed probabilities -- checked first, before any tier logic below.
 *
 * $modality (nullable): no MRI source with hemorrhage was available, so the model was trained and evaluated on
 * hemorrhage on CT only. A hemorrhagic prediction on MRI is therefore outside the evaluated scope and is shown as a
 * distinct "not validated" tier with no alert, never as a high- or moderate-risk finding. (On the held-out MRI cohorts
 * this occurred for 0 of 3,975 slices under the class-weighted rule and 1 under plain argmax.)
 */
function compute_risk_tier(string $prediction, float $probNormal, float $probIschemic, float $probHemorrhagic, ?bool $likelyNotAScan = null, ?string $modality = null): array
{
    if ($likelyNotAScan) {
        return ['level' => 'unreliable', 'label' => 'Unreliable — doesn\'t look like a scan', 'color' => '#6b3fa0', 'alert' => false];
    }

    if ($modality === 'MRI' && $prediction === 'hemorrhagic') {
        return ['level' => 'not_validated', 'label' => 'Not validated — hemorrhage on MRI is outside the evaluated scope', 'color' => '#7f8c8d', 'alert' => false];
    }

    if ($prediction === 'hemorrhagic') {
        if ($probHemorrhagic >= 0.55) {
            return ['level' => 'high', 'label' => 'High risk', 'color' => '#c0392b', 'alert' => true];
        }
        return ['level' => 'moderate', 'label' => 'Moderate risk', 'color' => '#e67e22', 'alert' => false];
    }

    if ($prediction === 'ischemic') {
        if ($probIschemic >= 0.90) {
            return ['level' => 'high', 'label' => 'High risk', 'color' => '#c0392b', 'alert' => true];
        }
        if ($probIschemic >= 0.70) {
            return ['level' => 'moderate', 'label' => 'Moderate risk', 'color' => '#e67e22', 'alert' => false];
        }
        return ['level' => 'uncertain', 'label' => 'Uncertain — recommend re-examine', 'color' => '#7f8c8d', 'alert' => false];
    }

    if ($prediction === 'normal') {
        if ($probNormal >= 0.80) {
            return ['level' => 'normal', 'label' => 'Normal', 'color' => '#2980b9', 'alert' => false];
        }
        return ['level' => 'uncertain', 'label' => 'Uncertain — recommend re-examine', 'color' => '#7f8c8d', 'alert' => false];
    }

    return ['level' => 'uncertain', 'label' => 'Uncertain', 'color' => '#7f8c8d', 'alert' => false];
}
