<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/audit.php';

// Mirrors review_form.php's POST branch: a finalized review is never edited
// in place -- a correction is recorded as a new row referencing the
// original (amends_review_id), which is preserved unmodified.

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_require_role(['doctor', 'radiologist']);

$body = api_json_body();
$uuid = (string)($body['scan_uuid'] ?? '');
if ($uuid === '') {
    api_json_error(400, 'Missing scan_uuid.');
}

$stmt = db()->prepare('SELECT * FROM scans WHERE public_uuid = :uuid');
$stmt->execute(['uuid' => $uuid]);
$scan = $stmt->fetch();
if (!$scan) {
    api_json_error(404, 'Scan not found.');
}
if (!provider_has_patient_access($user['id'], (int)$scan['patient_id'])) {
    api_json_error(403, 'You do not have permission to review this patient.');
}
if ($scan['status'] !== 'completed') {
    api_json_error(400, 'This scan has no completed AI result yet.');
}

$agrees = $body['agrees_with_ai'] ?? null;
$agreesValue = $agrees === true || $agrees === 1 || $agrees === '1' ? 1
    : ($agrees === false || $agrees === 0 || $agrees === '0' ? 0 : null);
$override = in_array($body['diagnosis_override'] ?? '', ['normal', 'ischemic', 'hemorrhagic', 'other', 'inconclusive'], true)
    ? $body['diagnosis_override'] : null;
$severity = in_array($body['severity'] ?? '', ['low', 'moderate', 'high', 'critical'], true)
    ? $body['severity'] : null;
$notes = trim((string)($body['notes'] ?? ''));
$finalize = (bool)($body['finalize'] ?? false);
$status = $finalize ? 'final' : 'draft';

$stmt = db()->prepare(
    'SELECT * FROM reviews WHERE scan_id = :sid AND reviewer_id = :rid ORDER BY id DESC LIMIT 1'
);
$stmt->execute(['sid' => $scan['id'], 'rid' => $user['id']]);
$myLatest = $stmt->fetch();

$isCorrection = $myLatest && $myLatest['status'] === 'final';
$editable = !$myLatest || $myLatest['status'] === 'draft';

if ($editable && $myLatest) {
    $stmt = db()->prepare(
        'UPDATE reviews SET agrees_with_ai = :agrees, diagnosis_override = :override, severity = :severity,
                             notes = :notes, status = :status
         WHERE id = :id'
    );
    $stmt->execute([
        'agrees' => $agreesValue, 'override' => $override, 'severity' => $severity,
        'notes' => $notes !== '' ? $notes : null, 'status' => $status, 'id' => $myLatest['id'],
    ]);
    $reviewId = (int)$myLatest['id'];
} else {
    $amends = $isCorrection ? $myLatest['id'] : null;
    $stmt = db()->prepare(
        'INSERT INTO reviews (scan_id, reviewer_id, reviewer_role, agrees_with_ai, diagnosis_override, severity, notes, status, amends_review_id)
         VALUES (:sid, :rid, :role, :agrees, :override, :severity, :notes, :status, :amends)'
    );
    $stmt->execute([
        'sid' => $scan['id'], 'rid' => $user['id'], 'role' => $user['role'],
        'agrees' => $agreesValue, 'override' => $override, 'severity' => $severity,
        'notes' => $notes !== '' ? $notes : null, 'status' => $status, 'amends' => $amends,
    ]);
    $reviewId = (int)db()->lastInsertId();
    log_action('review_created', 'review', (string)$reviewId, ['scan_id' => $scan['id'], 'via' => 'api']);
    if ($amends) {
        log_action('review_amended', 'review', (string)$reviewId, ['amends_review_id' => $amends, 'via' => 'api']);
    }
}

if ($status === 'final') {
    log_action('review_finalized', 'review', (string)$reviewId, ['scan_id' => $scan['id'], 'via' => 'api']);
}

api_send(['review_id' => $reviewId, 'status' => $status]);
