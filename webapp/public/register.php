<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';
require_once __DIR__ . '/includes/rate_limit.php';

bootstrap_session();

if (is_logged_in()) {
    header('Location: index.php');
    exit;
}

$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    csrf_check();
    $fullName = trim((string)($_POST['full_name'] ?? ''));
    $email = trim((string)($_POST['email'] ?? ''));
    $password = (string)($_POST['password'] ?? '');
    $passwordConfirm = (string)($_POST['password_confirm'] ?? '');
    $consent = isset($_POST['consent']);

    if (ip_registration_throttled()) {
        $error = 'Too many new accounts from this network. Please try again later.';
    } elseif ($fullName === '' || $email === '' || $password === '') {
        $error = 'Please fill in all fields.';
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error = 'Please enter a valid email address.';
    } elseif (strlen($password) < 8) {
        $error = 'Password must be at least 8 characters.';
    } elseif ($password !== $passwordConfirm) {
        $error = 'Passwords do not match.';
    } elseif (!$consent) {
        $error = 'You must acknowledge that this tool is not a diagnostic device to continue.';
    } else {
        $stmt = db()->prepare('SELECT id FROM users WHERE email = :email');
        $stmt->execute(['email' => $email]);
        if ($stmt->fetch()) {
            $error = 'An account with that email already exists.';
        } else {
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
            log_action('user_created', 'user', (string)$userId, ['self_registered' => true]);
            header('Location: dashboard_patient.php');
            exit;
        }
    }
}

$pageTitle = 'Register';
$narrowPage = true;
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan</span>
  <h1>Create a patient account</h1>
  <p class="lede">Doctor and radiologist accounts are set up by an administrator, not self-registered.</p>
</header>
<div class="card">
  <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
  <form method="post" action="register.php">
    <?= csrf_field() ?>
    <label for="full_name">Full name</label>
    <input type="text" id="full_name" name="full_name" required value="<?= htmlspecialchars($_POST['full_name'] ?? '', ENT_QUOTES) ?>">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required value="<?= htmlspecialchars($_POST['email'] ?? '', ENT_QUOTES) ?>">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required minlength="8">
    <div class="field-hint">At least 8 characters.</div>
    <label for="password_confirm">Confirm password</label>
    <input type="password" id="password_confirm" name="password_confirm" required minlength="8">
    <label style="display:flex; align-items:flex-start; gap:8px; font-weight:400; margin-top:18px;">
      <input type="checkbox" name="consent" style="width:auto; margin-top:3px;">
      <span>I understand NeuroScan is a research prototype: its AI predictions are not a medical diagnosis, are not for clinical use, and must be confirmed by a licensed radiologist or doctor.</span>
    </label>
    <div style="margin-top:20px;"><button type="submit">Create account</button></div>
  </form>
</div>
<p class="lede">Already have an account? <a href="login.php">Log in</a>.</p>
<?php require __DIR__ . '/includes/layout_footer.php'; ?>
