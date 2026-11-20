<?php
require_once __DIR__ . '/db.php';

function log_action(string $action, ?string $targetType = null, ?string $targetId = null, array $details = []): void
{
    $actorId = $_SESSION['user']['id'] ?? null;
    $actorRole = $_SESSION['user']['role'] ?? null;

    // Audit logging is a side effect of the real request, not its purpose --
    // a failure here (e.g. a stale session referencing a since-removed user)
    // must never break the action the user actually came to perform.
    try {
        $stmt = db()->prepare(
            'INSERT INTO audit_log (actor_user_id, actor_role, action, target_type, target_id, ip_address, user_agent, details)
             VALUES (:actor_id, :actor_role, :action, :target_type, :target_id, :ip, :ua, :details)'
        );
        $stmt->execute([
            'actor_id' => $actorId,
            'actor_role' => $actorRole,
            'action' => $action,
            'target_type' => $targetType,
            'target_id' => $targetId,
            'ip' => $_SERVER['REMOTE_ADDR'] ?? null,
            'ua' => substr($_SERVER['HTTP_USER_AGENT'] ?? '', 0, 255),
            'details' => $details ? json_encode($details) : null,
        ]);
    } catch (PDOException $e) {
        error_log('log_action failed: ' . $e->getMessage());
    }
}
