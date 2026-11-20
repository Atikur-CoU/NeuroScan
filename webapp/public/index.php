<?php
require_once __DIR__ . '/includes/auth.php';
bootstrap_session();

$user = current_user();
if (!$user) {
    header('Location: login.php');
    exit;
}
if (!empty($user['must_change_password'])) {
    header('Location: change_password.php');
    exit;
}

switch ($user['role']) {
    case 'patient':
        header('Location: dashboard_patient.php');
        break;
    case 'admin':
        header('Location: dashboard_admin.php');
        break;
    case 'doctor':
    case 'radiologist':
        header('Location: dashboard_provider.php');
        break;
}
exit;
