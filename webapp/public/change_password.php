<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();
$user = require_login();

$error = null;
$forced = !empty($user['must_change_password']);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    csrf_check();
    $current = (string)($_POST['current_password'] ?? '');
    $new = (string)($_POST['new_password'] ?? '');
    $confirm = (string)($_POST['new_password_confirm'] ?? '');

    $stmt = db()->prepare('SELECT password_hash FROM users WHERE id = :id');
    $stmt->execute(['id' => $user['id']]);
    $row = $stmt->fetch();

    if (!$row || !password_verify($current, $row['password_hash'])) {
        $error = 'Current password is incorrect.';
    } elseif (strlen($new) < 8) {
        $error = 'New password must be at least 8 characters.';
    } elseif ($new !== $confirm) {
        $error = 'New passwords do not match.';
    } else {
        $stmt = db()->prepare(
            'UPDATE users SET password_hash = :hash, must_change_password = 0 WHERE id = :id'
        );
        $stmt->execute(['hash' => password_hash($new, PASSWORD_DEFAULT), 'id' => $user['id']]);
        clear_must_change_password();
        log_action('password_changed', 'user', (string)$user['id']);
        header('Location: index.php');
        exit;
    }
}

$pageTitle = 'Change password';
$narrowPage = true;
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan</span>
  <h1>Change password</h1>
  <?php if ($forced): ?>
    <p class="lede">Your account was created with a temporary password. Set your own password to continue.</p>
  <?php endif; ?>
</header>
<div class="card">
  <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
  <form method="post" action="change_password.php">
    <?= csrf_field() ?>
    <label for="current_password">Current password</label>
    <input type="password" id="current_password" name="current_password" required autofocus>
    <label for="new_password">New password</label>
    <input type="password" id="new_password" name="new_password" required minlength="8">
    <label for="new_password_confirm">Confirm new password</label>
    <input type="password" id="new_password_confirm" name="new_password_confirm" required minlength="8">
    <div style="margin-top:20px;"><button type="submit">Update password</button></div>
  </form>
</div>
<?php require __DIR__ . '/includes/layout_footer.php'; ?>
