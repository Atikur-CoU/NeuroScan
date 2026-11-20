<?php
require_once __DIR__ . '/auth.php';

// Bearer-token auth for the Android client. Deliberately reuses the same
// session-based user model as the web app (login_user(), current_user(),
// log_action(), provider_has_patient_access()) so those functions work
// identically for a token-authenticated API request without modification --
// each API request bootstraps a fresh, throwaway PHP session and populates
// it from the resolved token's owner, exactly like a password login would.

const API_TOKEN_TTL_DAYS = 90;

function api_json_error(int $httpCode, string $message): void
{
    http_response_code($httpCode);
    header('Content-Type: application/json');
    echo json_encode(['error' => $message]);
    exit;
}

function api_send(array $data, int $httpCode = 200): void
{
    http_response_code($httpCode);
    header('Content-Type: application/json');
    echo json_encode($data);
    exit;
}

/** Issues a new opaque bearer token for a user. Only the SHA-256 hash is stored. */
function api_issue_token(int $userId, ?string $deviceLabel = null): string
{
    $token = bin2hex(random_bytes(32));
    $expiresAt = gmdate('Y-m-d H:i:s', time() + API_TOKEN_TTL_DAYS * 86400);
    $stmt = db()->prepare(
        'INSERT INTO api_tokens (user_id, token_hash, device_label, expires_at) VALUES (:uid, :hash, :label, :exp)'
    );
    $stmt->execute([
        'uid' => $userId,
        'hash' => hash('sha256', $token),
        'label' => $deviceLabel,
        'exp' => $expiresAt,
    ]);
    return $token;
}

function api_bearer_token_from_request(): ?string
{
    $header = $_SERVER['HTTP_AUTHORIZATION'] ?? '';
    if ($header === '' && function_exists('apache_request_headers')) {
        // Some SAPIs don't populate HTTP_AUTHORIZATION unless configured to.
        foreach (apache_request_headers() as $k => $v) {
            if (strcasecmp($k, 'Authorization') === 0) {
                $header = $v;
                break;
            }
        }
    }
    if (preg_match('/^Bearer\s+([A-Fa-f0-9]{64})$/', trim($header), $m)) {
        return $m[1];
    }
    return null;
}

/** Authenticates the current request's bearer token. Exits with a 401 JSON error on failure. */
function api_authenticate(): array
{
    bootstrap_session();
    $token = api_bearer_token_from_request();
    if ($token === null) {
        api_json_error(401, 'Missing or malformed Authorization header.');
    }

    $stmt = db()->prepare(
        'SELECT t.id AS token_id, u.*
         FROM api_tokens t JOIN users u ON u.id = t.user_id
         WHERE t.token_hash = :hash AND t.expires_at > UTC_TIMESTAMP()'
    );
    $stmt->execute(['hash' => hash('sha256', $token)]);
    $row = $stmt->fetch();

    if (!$row || !$row['is_active']) {
        api_json_error(401, 'Invalid, expired, or revoked token.');
    }

    $stmt = db()->prepare('UPDATE api_tokens SET last_used_at = UTC_TIMESTAMP() WHERE id = :id');
    $stmt->execute(['id' => $row['token_id']]);

    login_user($row);
    $_SESSION['user']['api_token_id'] = (int)$row['token_id'];

    // Same rule as require_login() on the web: an account with a temporary password reaches nothing until the
    // password is changed (on the website); only who-am-I and logout stay open, so the app can tell the user.
    if (!empty($row['must_change_password'])
        && !in_array(basename($_SERVER['SCRIPT_NAME'] ?? ''), ['me.php', 'logout.php'], true)) {
        api_json_error(403, 'Password change required. Log in on the website and set a new password first.');
    }
    return current_user();
}

/** Authenticates and requires the resolved user's role be one of $roles. Exits with a 403 JSON error otherwise. */
function api_require_role(array $roles): array
{
    $user = api_authenticate();
    if (!in_array($user['role'], $roles, true)) {
        api_json_error(403, 'You do not have permission to perform this action.');
    }
    return $user;
}

function api_revoke_current_token(): void
{
    $user = current_user();
    if ($user && isset($user['api_token_id'])) {
        $stmt = db()->prepare('DELETE FROM api_tokens WHERE id = :id');
        $stmt->execute(['id' => $user['api_token_id']]);
    }
}

/** Reads and JSON-decodes the raw POST body -- for endpoints receiving application/json rather than form/multipart. */
function api_json_body(): array
{
    $raw = file_get_contents('php://input');
    $data = json_decode((string)$raw, true);
    return is_array($data) ? $data : [];
}
