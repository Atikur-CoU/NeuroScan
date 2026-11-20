<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/result_gate.php';

// Role-aware scan list, mirroring dashboard_patient.php / dashboard_provider.php / provider_patient.php:
//   - patient:  their own scans.
//   - provider: ?pending_review=1 -> completed scans from assigned patients they haven't reviewed yet
//               (dashboard_provider.php's "Pending my review"); ?patient_id=N -> one assigned
//               patient's full history (provider_patient.php); no filter -> same as pending_review=1.

if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_authenticate();

if ($user['role'] === 'patient') {
    $stmt = db()->prepare(
        "SELECT s.public_uuid, s.modality, s.status, s.uploaded_at, s.notes,
                p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic,
                (SELECT COUNT(*) FROM reviews r WHERE r.scan_id = s.id AND r.status = 'final') AS final_review_count
         FROM scans s
         LEFT JOIN predictions p ON p.scan_id = s.id
         WHERE s.patient_id = :pid
         ORDER BY s.uploaded_at DESC"
    );
    $stmt->execute(['pid' => $user['id']]);
    $gate = result_gate_enabled();
    $scans = [];
    foreach ($stmt->fetchAll() as $s) {
        $hidden = $gate && (int)$s['final_review_count'] === 0;
        if ($hidden) {
            $s['prediction'] = $s['prob_normal'] = $s['prob_ischemic'] = $s['prob_hemorrhagic'] = null;
        }
        unset($s['final_review_count']);
        $s['awaiting_review'] = $hidden && $s['status'] === 'completed';
        $scans[] = $s;
    }
    api_send(['scans' => $scans]);
}

if (!in_array($user['role'], ['doctor', 'radiologist'], true)) {
    api_json_error(403, 'You do not have permission to perform this action.');
}

$patientId = isset($_GET['patient_id']) ? (int)$_GET['patient_id'] : null;

if ($patientId !== null) {
    if (!provider_has_patient_access($user['id'], $patientId)) {
        api_json_error(403, 'You do not have permission to view this patient.');
    }
    $stmt = db()->prepare(
        "SELECT s.public_uuid, s.modality, s.status, s.uploaded_at, p.prediction,
                (SELECT COUNT(*) FROM reviews WHERE scan_id = s.id AND status = 'final') AS review_count
         FROM scans s
         LEFT JOIN predictions p ON p.scan_id = s.id
         WHERE s.patient_id = :pid
         ORDER BY s.uploaded_at DESC"
    );
    $stmt->execute(['pid' => $patientId]);
    api_send(['scans' => $stmt->fetchAll()]);
}

// Default / ?pending_review=1: completed scans from assigned patients this provider hasn't reviewed yet.
$stmt = db()->prepare(
    "SELECT s.public_uuid, s.modality, s.uploaded_at, p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic,
            u.id AS patient_id, u.full_name AS patient_name
     FROM scans s
     JOIN predictions p ON p.scan_id = s.id
     JOIN users u ON u.id = s.patient_id
     JOIN doctor_patient dp ON dp.patient_id = s.patient_id AND dp.provider_id = :provider_id AND dp.is_active = 1
     LEFT JOIN reviews r ON r.scan_id = s.id AND r.reviewer_id = :provider_id2
     WHERE s.status = 'completed' AND r.id IS NULL
     ORDER BY s.uploaded_at ASC"
);
$stmt->execute(['provider_id' => $user['id'], 'provider_id2' => $user['id']]);
api_send(['scans' => $stmt->fetchAll()]);
