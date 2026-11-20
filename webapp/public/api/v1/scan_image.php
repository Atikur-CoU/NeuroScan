<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/result_gate.php';

// Streams the stored image, same ownership check as scan_image.php. The
// Android client requests this with an Authorization header (via its HTTP
// image-loading library), not a token in the URL, so no credential ever
// appears in a log line or is cacheable by an intermediary as a URL.

$user = api_authenticate();

$uuid = (string)($_GET['uuid'] ?? '');
$type = ($_GET['type'] ?? 'original') === 'heatmap' ? 'heatmap' : 'original';
if ($uuid === '') {
    http_response_code(404);
    exit;
}

$stmt = db()->prepare(
    'SELECT s.id, s.patient_id, s.storage_path, s.mime_type, p.heatmap_path
     FROM scans s LEFT JOIN predictions p ON p.scan_id = s.id
     WHERE s.public_uuid = :uuid'
);
$stmt->execute(['uuid' => $uuid]);
$scan = $stmt->fetch();

if (!$scan) {
    http_response_code(404);
    exit;
}

$isOwner = $user['role'] === 'patient' && (int)$scan['patient_id'] === (int)$user['id'];
$isProvider = in_array($user['role'], ['doctor', 'radiologist'], true)
    && provider_has_patient_access($user['id'], (int)$scan['patient_id']);
$isAdmin = $user['role'] === 'admin';
if (!$isOwner && !$isProvider && !$isAdmin) {
    http_response_code(403);
    exit;
}

if ($type === 'heatmap') {
    if (result_hidden_from_user($user, (int)$scan['id'])) {
        http_response_code(403);
        exit;
    }
    if (!$scan['heatmap_path']) {
        http_response_code(404);
        exit;
    }
    $relativePath = $scan['heatmap_path'];
    $mimeType = 'image/png';
} else {
    $relativePath = $scan['storage_path'];
    $mimeType = $scan['mime_type'];
}

$fullPath = UPLOADS_DIR . '/' . $relativePath;
$realUploads = realpath(UPLOADS_DIR);
$realFile = realpath($fullPath);
if ($realFile === false || $realUploads === false || strpos($realFile, $realUploads) !== 0 || !is_file($realFile)) {
    http_response_code(404);
    exit;
}

header('Content-Type: ' . $mimeType);
header('Content-Length: ' . filesize($realFile));
header('Cache-Control: private, max-age=3600');
readfile($realFile);
