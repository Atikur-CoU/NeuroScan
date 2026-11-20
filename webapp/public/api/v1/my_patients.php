<?php
require_once __DIR__ . '/../../includes/api_auth.php';

// Mirrors dashboard_provider.php's "My patients" query.

if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_require_role(['doctor', 'radiologist']);

$stmt = db()->prepare(
    "SELECT u.id, u.full_name, u.email, dp.relationship_type,
            (SELECT COUNT(*) FROM scans WHERE patient_id = u.id) AS scan_count
     FROM doctor_patient dp
     JOIN users u ON u.id = dp.patient_id
     WHERE dp.provider_id = :provider_id AND dp.is_active = 1
     ORDER BY u.full_name"
);
$stmt->execute(['provider_id' => $user['id']]);

api_send(['patients' => $stmt->fetchAll()]);
