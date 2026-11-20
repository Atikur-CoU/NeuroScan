<?php
require_once __DIR__ . '/inference_client.php';

// Clinician approval gate: when the 'gate_patient_results' setting is '1', a
// patient sees no AI result (class, probabilities, alert tier or heatmap) for
// a scan until a doctor or radiologist has finalized a review of it. Checked
// on the server for the web pages and the API alike, so no client can show an
// unapproved result. Off by default (schema seed and migration set '0').

function result_gate_enabled(): bool
{
    return get_setting('gate_patient_results', '0') === '1';
}

function scan_has_final_review(int $scanId): bool
{
    $stmt = db()->prepare("SELECT 1 FROM reviews WHERE scan_id = :sid AND status = 'final' LIMIT 1");
    $stmt->execute(['sid' => $scanId]);
    return (bool)$stmt->fetch();
}

function result_hidden_from_user(array $user, int $scanId): bool
{
    return $user['role'] === 'patient' && result_gate_enabled() && !scan_has_final_review($scanId);
}
