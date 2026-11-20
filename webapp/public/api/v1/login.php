<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/rate_limit.php';
require_once __DIR__ . '/../../includes/audit.php';

bootstrap_session();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    api_json_error(405, 'Method not allowed.');
}

$body = api_json_body();
$email = trim((string)($body['email'] ?? ''));
$password = (string)($body['password'] ?? '');
$deviceLabel = trim((string)($body['device_label'] ?? 'Android'));

if ($email === '' || $password === '') {
    api_json_error(400, 'Email and password are required.');
}
if (is_locked_out($email) || ip_login_throttled()) {
    api_json_error(429, 'Too many failed attempts. Please try again in a few minutes.');
}

$stmt = db()->prepare('SELECT * FROM users WHERE email = :email');
$stmt->execute(['email' => $email]);
$row = $stmt->fetch();

if (!$row || !$row['is_active'] || !password_verify($password, $row['password_hash'])) {
    if ($row) {
        register_failed_login($email);
    }
    record_login_attempt($email, false);
    log_action('login_failed', 'user', null, ['email' => $email, 'via' => 'api']);
    api_json_error(401, 'Invalid email or password.');
}

clear_failed_logins($email);
record_login_attempt($email, true);
login_user($row);
$token = api_issue_token((int)$row['id'], $deviceLabel !== '' ? substr($deviceLabel, 0, 100) : 'Android');
log_action('login_success', 'user', (string)$row['id'], ['via' => 'api']);

api_send([
    'token' => $token,
    'user' => [
        'id' => (int)$row['id'],
        'role' => $row['role'],
        'full_name' => $row['full_name'],
        'email' => $row['email'],
        'must_change_password' => (bool)$row['must_change_password'],
    ],
]);
