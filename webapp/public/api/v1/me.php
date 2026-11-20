<?php
require_once __DIR__ . '/../../includes/api_auth.php';

if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_authenticate();
api_send([
    'id' => $user['id'],
    'role' => $user['role'],
    'full_name' => $user['full_name'],
    'email' => $user['email'],
    'must_change_password' => (bool)$user['must_change_password'],
]);
