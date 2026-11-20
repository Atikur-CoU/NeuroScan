<?php
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/db.php';

function bootstrap_session(): void
{
    if (session_status() === PHP_SESSION_ACTIVE) {
        return;
    }
    session_set_cookie_params([
        'lifetime' => 0,
        'path' => '/',
        'httponly' => true,
        'secure' => FORCE_HTTPS_COOKIES,
        'samesite' => 'Lax',
    ]);
    session_start();
}

function current_user(): ?array
{
    return $_SESSION['user'] ?? null;
}

function is_logged_in(): bool
{
    return current_user() !== null;
}

function login_user(array $userRow): void
{
    session_regenerate_id(true);
    $_SESSION['user'] = [
        'id' => (int)$userRow['id'],
        'role' => $userRow['role'],
        'full_name' => $userRow['full_name'],
        'email' => $userRow['email'],
        'must_change_password' => (bool)$userRow['must_change_password'],
    ];
}

function clear_must_change_password(): void
{
    if (isset($_SESSION['user'])) {
        $_SESSION['user']['must_change_password'] = false;
    }
}

function logout_user(): void
{
    $_SESSION = [];
    if (ini_get('session.use_cookies')) {
        $params = session_get_cookie_params();
        setcookie(session_name(), '', time() - 42000, $params['path'], $params['domain'], $params['secure'], $params['httponly']);
    }
    session_destroy();
}

const PASSWORD_CHANGE_EXEMPT_SCRIPTS = ['change_password.php', 'logout.php'];

function require_login(): array
{
    if (!is_logged_in()) {
        header('Location: login.php');
        exit;
    }
    $user = current_user();
    $currentScript = basename($_SERVER['SCRIPT_NAME'] ?? '');
    if (!empty($user['must_change_password']) && !in_array($currentScript, PASSWORD_CHANGE_EXEMPT_SCRIPTS, true)) {
        header('Location: change_password.php');
        exit;
    }
    return $user;
}

function require_role(array $roles): array
{
    $user = require_login();
    if (!in_array($user['role'], $roles, true)) {
        http_response_code(403);
        die('You do not have permission to view this page.');
    }
    return $user;
}

/** Whether a doctor/radiologist has an active doctor_patient link to a patient. */
function provider_has_patient_access(int $providerId, int $patientId): bool
{
    $stmt = db()->prepare(
        'SELECT 1 FROM doctor_patient WHERE provider_id = :prid AND patient_id = :pid AND is_active = 1 LIMIT 1'
    );
    $stmt->execute(['prid' => $providerId, 'pid' => $patientId]);
    return (bool)$stmt->fetch();
}
