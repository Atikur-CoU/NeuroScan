<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/audit.php';
require_once __DIR__ . '/../../includes/risk_tier.php';
require_once __DIR__ . '/../../includes/inference_client.php';
require_once __DIR__ . '/../../includes/result_gate.php';

// Mirrors scan_view2.php + review_form.php's fetch logic: scan + prediction +
// patient name + finalized reviews, plus (for a provider) their own latest
// review row (draft or final) on this scan.

if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_authenticate();

$uuid = (string)($_GET['uuid'] ?? '');
if ($uuid === '') {
    api_json_error(400, 'Missing uuid.');
}

$stmt = db()->prepare(
    'SELECT s.*, p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic, p.heatmap_path, p.likely_not_a_scan, p.last_error,
            p.pathway, p.evaluated_pathway, p.warnings, p.decision_note, p.model_version,
            u.full_name AS patient_name
     FROM scans s
     LEFT JOIN predictions p ON p.scan_id = s.id
     JOIN users u ON u.id = s.patient_id
     WHERE s.public_uuid = :uuid'
);
$stmt->execute(['uuid' => $uuid]);
$scan = $stmt->fetch();

if (!$scan) {
    api_json_error(404, 'Scan not found.');
}

$isOwner = $user['role'] === 'patient' && (int)$scan['patient_id'] === (int)$user['id'];
$isProvider = in_array($user['role'], ['doctor', 'radiologist'], true)
    && provider_has_patient_access($user['id'], (int)$scan['patient_id']);
$isAdmin = $user['role'] === 'admin';
if (!$isOwner && !$isProvider && !$isAdmin) {
    api_json_error(403, 'You do not have permission to view this scan.');
}

log_action('scan_viewed', 'scan', (string)$scan['id'], ['via' => 'api']);

$stmt = db()->prepare(
    "SELECT r.*, u.full_name AS reviewer_name
     FROM reviews r JOIN users u ON u.id = r.reviewer_id
     WHERE r.scan_id = :sid AND r.status = 'final'
     ORDER BY r.created_at DESC"
);
$stmt->execute(['sid' => $scan['id']]);
$finalReviews = $stmt->fetchAll();

$resultHidden = result_hidden_from_user($user, (int)$scan['id']);

$myLatestReview = null;
if ($isProvider) {
    $stmt = db()->prepare(
        'SELECT * FROM reviews WHERE scan_id = :sid AND reviewer_id = :rid ORDER BY id DESC LIMIT 1'
    );
    $stmt->execute(['sid' => $scan['id'], 'rid' => $user['id']]);
    $myLatestReview = $stmt->fetch() ?: null;
}

api_send([
    'scan' => [
        'public_uuid' => $scan['public_uuid'],
        'modality' => $scan['modality'],
        'status' => $scan['status'],
        'notes' => $scan['notes'],
        'uploaded_at' => $scan['uploaded_at'],
        'original_filename' => $scan['original_filename'],
        'patient_id' => (int)$scan['patient_id'],
        'patient_name' => $scan['patient_name'],
    ],
    'awaiting_review' => $resultHidden && $scan['status'] === 'completed',
    'prediction' => $scan['status'] === 'completed' && !$resultHidden ? array_merge([
        'prediction' => $scan['prediction'],
        'prob_normal' => (float)$scan['prob_normal'],
        'prob_ischemic' => (float)$scan['prob_ischemic'],
        'prob_hemorrhagic' => (float)$scan['prob_hemorrhagic'],
        'heatmap_available' => $scan['heatmap_path'] !== null,
        'likely_not_a_scan' => $scan['likely_not_a_scan'] === null ? null : (bool)$scan['likely_not_a_scan'],
        'pathway' => $scan['pathway'],
        'evaluated_pathway' => $scan['evaluated_pathway'] === null ? null : (bool)$scan['evaluated_pathway'],
        'warnings' => prediction_warnings($scan['warnings']),
        'decision_note' => $scan['decision_note'],
        'model_version' => $scan['model_version'],
    ], compute_risk_tier(
        (string)$scan['prediction'],
        (float)$scan['prob_normal'],
        (float)$scan['prob_ischemic'],
        (float)$scan['prob_hemorrhagic'],
        $scan['likely_not_a_scan'] !== null && (bool)$scan['likely_not_a_scan'],
        (string)$scan['modality']
    ), $isOwner && result_gate_enabled() ? ['alert' => false] : []) : null,
    'error' => $scan['status'] === 'failed' ? $scan['last_error'] : null,
    'final_reviews' => $finalReviews,
    'my_latest_review' => $myLatestReview,
]);
