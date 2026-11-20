<?php
require_once __DIR__ . '/db.php';
require_once __DIR__ . '/audit.php';

const LOGIN_MAX_ATTEMPTS = 5;
const LOGIN_WINDOW_MINUTES = 15;
const LOGIN_LOCKOUT_MINUTES = 15;

// Per-address limits, on top of the per-account lockout: they stop one address trying a few passwords on many
// accounts (password spraying) or creating accounts in bulk. Generous enough for a shared campus or NAT address.
const IP_MAX_FAILED_LOGINS = 20;
const IP_MAX_REGISTRATIONS_PER_HOUR = 5;

function ip_login_throttled(): bool
{
    $stmt = db()->prepare(
        'SELECT COUNT(*) AS c FROM login_attempts
         WHERE ip_address = :ip AND success = 0 AND attempted_at > NOW() - INTERVAL ' . LOGIN_WINDOW_MINUTES . ' MINUTE'
    );
    $stmt->execute(['ip' => $_SERVER['REMOTE_ADDR'] ?? '']);
    return (int)$stmt->fetch()['c'] >= IP_MAX_FAILED_LOGINS;
}

function ip_registration_throttled(): bool
{
    $stmt = db()->prepare(
        "SELECT COUNT(*) AS c FROM audit_log
         WHERE action = 'user_created' AND ip_address = :ip AND actor_role = 'patient'
           AND created_at > NOW() - INTERVAL 1 HOUR"
    );
    $stmt->execute(['ip' => $_SERVER['REMOTE_ADDR'] ?? '']);
    return (int)$stmt->fetch()['c'] >= IP_MAX_REGISTRATIONS_PER_HOUR;
}

function record_login_attempt(string $email, bool $success): void
{
    $stmt = db()->prepare(
        'INSERT INTO login_attempts (email, ip_address, success) VALUES (:email, :ip, :success)'
    );
    $stmt->execute([
        'email' => $email,
        'ip' => $_SERVER['REMOTE_ADDR'] ?? '',
        'success' => $success ? 1 : 0,
    ]);
}

function is_locked_out(string $email): bool
{
    $stmt = db()->prepare('SELECT locked_until FROM users WHERE email = :email');
    $stmt->execute(['email' => $email]);
    $row = $stmt->fetch();
    if (!$row || !$row['locked_until']) {
        return false;
    }
    return strtotime($row['locked_until'] . ' UTC') > time();
}

function register_failed_login(string $email): void
{
    $stmt = db()->prepare(
        'UPDATE users SET failed_login_count = failed_login_count + 1 WHERE email = :email'
    );
    $stmt->execute(['email' => $email]);

    $stmt = db()->prepare('SELECT failed_login_count FROM users WHERE email = :email');
    $stmt->execute(['email' => $email]);
    $row = $stmt->fetch();

    if ($row && (int)$row['failed_login_count'] >= LOGIN_MAX_ATTEMPTS) {
        $lockUntil = gmdate('Y-m-d H:i:s', time() + LOGIN_LOCKOUT_MINUTES * 60);
        $stmt = db()->prepare('UPDATE users SET locked_until = :until WHERE email = :email');
        $stmt->execute(['until' => $lockUntil, 'email' => $email]);
        log_action('account_locked', 'user', null, ['email' => $email]);
    }
}

function clear_failed_logins(string $email): void
{
    $stmt = db()->prepare(
        'UPDATE users SET failed_login_count = 0, locked_until = NULL WHERE email = :email'
    );
    $stmt->execute(['email' => $email]);
}
