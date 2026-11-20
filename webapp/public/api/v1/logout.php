<?php
require_once __DIR__ . '/../../includes/api_auth.php';
require_once __DIR__ . '/../../includes/audit.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    api_json_error(405, 'Method not allowed.');
}

$user = api_authenticate();
api_revoke_current_token();
log_action('logout', 'user', (string)$user['id'], ['via' => 'api']);

api_send(['ok' => true]);
