<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/rate_limit.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();

if (is_logged_in()) {
    header('Location: index.php');
    exit;
}

$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    csrf_check();
    $email = trim((string)($_POST['email'] ?? ''));
    $password = (string)($_POST['password'] ?? '');

    if ($email === '' || $password === '') {
        $error = 'Please enter your email and password.';
    } elseif (is_locked_out($email) || ip_login_throttled()) {
        $error = 'Too many failed attempts. Please try again in a few minutes.';
    } else {
        $stmt = db()->prepare('SELECT * FROM users WHERE email = :email');
        $stmt->execute(['email' => $email]);
        $row = $stmt->fetch();

        if ($row && $row['is_active'] && password_verify($password, $row['password_hash'])) {
            clear_failed_logins($email);
            record_login_attempt($email, true);
            login_user($row);
            log_action('login_success');
            header('Location: index.php');
            exit;
        }

        if ($row) {
            register_failed_login($email);
        }
        record_login_attempt($email, false);
        log_action('login_failed', 'user', null, ['email' => $email]);
        $error = 'Invalid email or password.';
    }
}

$pageTitle = 'Log in';
$narrowPage = true;
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan</span>
  <h1>Log in</h1>
</header>
<div class="card">
  <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
  <form method="post" action="login.php">
    <?= csrf_field() ?>
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required autofocus value="<?= htmlspecialchars($_POST['email'] ?? '', ENT_QUOTES) ?>">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required>
    <div style="margin-top:20px;"><button type="submit">Log in</button></div>
  </form>
</div>
<p class="lede">No account? <a href="register.php">Register as a patient</a>.</p>
<?php require __DIR__ . '/includes/layout_footer.php'; ?>
