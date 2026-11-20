<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/audit.php';

bootstrap_session();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    api_json_error(405, 'Method not allowed.');
}

$body = api_json_body();
$fullName = trim((string)($body['full_name'] ?? ''));
$email = trim((string)($body['email'] ?? ''));
$password = (string)($body['password'] ?? '');
$passwordConfirm = (string)($body['password_confirm'] ?? '');
$consent = (bool)($body['consent'] ?? false);
$deviceLabel = trim((string)($body['device_label'] ?? 'Android'));

if ($fullName === '' || $email === '' || $password === '') {
    api_json_error(400, 'Please fill in all fields.');
}
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    api_json_error(400, 'Please enter a valid email address.');
}
if (strlen($password) < 8) {
    api_json_error(400, 'Password must be at least 8 characters.');
}
if ($password !== $passwordConfirm) {
    api_json_error(400, 'Passwords do not match.');
}
if (!$consent) {
    api_json_error(400, 'You must acknowledge that this tool is not a diagnostic device to continue.');
}

$stmt = db()->prepare('SELECT id FROM users WHERE email = :email');
$stmt->execute(['email' => $email]);
if ($stmt->fetch()) {
    api_json_error(409, 'An account with that email already exists.');
}

db()->beginTransaction();
$stmt = db()->prepare(
    'INSERT INTO users (role, full_name, email, password_hash) VALUES (:role, :name, :email, :hash)'
);
$stmt->execute([
    'role' => 'patient',
    'name' => $fullName,
    'email' => $email,
    'hash' => password_hash($password, PASSWORD_DEFAULT),
]);
$userId = (int)db()->lastInsertId();
$stmt = db()->prepare('INSERT INTO patient_profiles (user_id) VALUES (:id)');
$stmt->execute(['id' => $userId]);
db()->commit();

$stmt = db()->prepare('SELECT * FROM users WHERE id = :id');
$stmt->execute(['id' => $userId]);
$row = $stmt->fetch();

login_user($row);
$token = api_issue_token($userId, $deviceLabel !== '' ? substr($deviceLabel, 0, 100) : 'Android');
log_action('user_created', 'user', (string)$userId, ['self_registered' => true, 'via' => 'api']);

api_send([
    'token' => $token,
    'user' => [
        'id' => $userId,
        'role' => 'patient',
        'full_name' => $fullName,
        'email' => $email,
        'must_change_password' => false,
    ],
], 201);
