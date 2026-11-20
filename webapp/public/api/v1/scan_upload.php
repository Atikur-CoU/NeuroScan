<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/audit.php';
require_once __DIR__ . '/../../includes/upload_validate.php';
require_once __DIR__ . '/../../includes/inference_client.php';
require_once __DIR__ . '/../../includes/result_gate.php';

// Multipart upload, mirroring scan_upload.php's POST branch exactly
// (same validation, storage, transaction, and inference call). Optional
// context images: prev_file + next_file (CT) or adc_file (MRI).

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_require_role(['patient']);

$modality = ($_POST['modality'] ?? '') === 'MRI' ? 'MRI' : 'CT';
$notes = trim((string)($_POST['notes'] ?? ''));
$maxMb = (int)get_setting('max_upload_mb', '10');

if (empty($_FILES['file'])) {
    api_json_error(400, 'Please choose an image file.');
}

try {
    $stored = validate_and_store_upload($_FILES['file'], $maxMb * 1024 * 1024);
    $context = store_context_uploads($_FILES, $modality, $maxMb * 1024 * 1024);
} catch (UploadValidationError $e) {
    api_json_error(400, $e->getMessage());
}

$uuid = sprintf(
    '%08x-%04x-%04x-%04x-%012x',
    random_int(0, 0xffffffff),
    random_int(0, 0xffff),
    random_int(0, 0xffff) & 0x0fff | 0x4000,
    random_int(0, 0xffff) & 0x3fff | 0x8000,
    random_int(0, 0xffffffffffff)
);

db()->beginTransaction();
$stmt = db()->prepare(
    'INSERT INTO scans (public_uuid, patient_id, uploaded_by, modality, original_filename, stored_filename, storage_path, context_files, file_size_bytes, mime_type, sha256_hash, status, notes)
     VALUES (:uuid, :pid, :uid, :modality, :orig, :stored, :path, :context, :size, :mime, :hash, :status, :notes)'
);
$stmt->execute([
    'uuid' => $uuid,
    'pid' => $user['id'],
    'uid' => $user['id'],
    'modality' => $modality,
    'orig' => $_FILES['file']['name'],
    'stored' => $stored['stored_filename'],
    'path' => $stored['storage_path'],
    'context' => $context ? json_encode($context) : null,
    'size' => $stored['file_size_bytes'],
    'mime' => $stored['mime_type'],
    'hash' => $stored['sha256_hash'],
    'status' => 'processing',
    'notes' => $notes !== '' ? $notes : null,
]);
$scanId = (int)db()->lastInsertId();

$stmt = db()->prepare('INSERT INTO predictions (scan_id, request_started_at) VALUES (:sid, UTC_TIMESTAMP())');
$stmt->execute(['sid' => $scanId]);
db()->commit();

log_action('scan_uploaded', 'scan', (string)$scanId, ['via' => 'api']);

$scanRow = db()->prepare('SELECT * FROM scans WHERE id = :sid');
$scanRow->execute(['sid' => $scanId]);

try {
    $result = run_and_store_prediction($scanRow->fetch(), ['via' => 'api']);
    $oodFlag = likely_not_a_scan_value($result);

    if (result_gate_enabled()) {
        api_send(['public_uuid' => $uuid, 'status' => 'completed', 'awaiting_review' => true], 201);
    }

    api_send([
        'public_uuid' => $uuid,
        'status' => 'completed',
        'prediction' => $result['prediction'],
        'prob_normal' => (float)$result['prob_normal'],
        'prob_ischemic' => (float)$result['prob_ischemic'],
        'prob_hemorrhagic' => (float)$result['prob_hemorrhagic'],
        'likely_not_a_scan' => $oodFlag === null ? null : (bool)$oodFlag,
        'pathway' => $result['pathway'] ?? null,
        'evaluated_pathway' => isset($result['evaluated_pathway']) ? (bool)$result['evaluated_pathway'] : null,
        'warnings' => isset($result['warnings']) && is_array($result['warnings']) ? array_values($result['warnings']) : [],
        'decision_note' => $result['decision_note'] ?? null,
        'model_version' => $result['model_version'] ?? null,
    ], 201);
} catch (InferenceError $e) {
    api_send([
        'public_uuid' => $uuid,
        'status' => 'failed',
        'error' => $e->getMessage(),
    ], 201);
}
