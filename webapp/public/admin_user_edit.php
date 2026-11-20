<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();
$admin = require_role(['admin']);

$targetId = (int)($_GET['id'] ?? $_POST['id'] ?? 0);
$stmt = db()->prepare('SELECT * FROM users WHERE id = :id');
$stmt->execute(['id' => $targetId]);
$target = $stmt->fetch();

if (!$target) {
    http_response_code(404);
    die('User not found.');
}

$error = null;
$saved = false;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    csrf_check();
    $fullName = trim((string)($_POST['full_name'] ?? ''));
    $email = trim((string)($_POST['email'] ?? ''));
    $phone = trim((string)($_POST['phone'] ?? ''));
    $license = trim((string)($_POST['license_number'] ?? ''));

    if ($fullName === '' || $email === '') {
        $error = 'Name and email are required.';
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error = 'Please enter a valid email address.';
    } else {
        $stmt = db()->prepare('SELECT id FROM users WHERE email = :email AND id != :id');
        $stmt->execute(['email' => $email, 'id' => $targetId]);
        if ($stmt->fetch()) {
            $error = 'Another account already uses that email.';
        } else {
            $before = [
                'full_name' => $target['full_name'], 'email' => $target['email'],
                'phone' => $target['phone'], 'license_number' => $target['license_number'],
            ];
            $stmt = db()->prepare(
                'UPDATE users SET full_name = :name, email = :email, phone = :phone, license_number = :license WHERE id = :id'
            );
            $stmt->execute([
                'name' => $fullName, 'email' => $email,
                'phone' => $phone !== '' ? $phone : null,
                'license' => $license !== '' ? $license : null,
                'id' => $targetId,
            ]);
            log_action('user_updated', 'user', (string)$targetId, ['before' => $before]);
            $stmt = db()->prepare('SELECT * FROM users WHERE id = :id');
            $stmt->execute(['id' => $targetId]);
            $target = $stmt->fetch();
            $saved = true;
        }
    }
}

$pageTitle = 'Edit user';
$narrowPage = true;
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Edit <?= htmlspecialchars($target['full_name'], ENT_QUOTES) ?></h1>
  <p class="lede">Role: <?= htmlspecialchars($target['role'], ENT_QUOTES) ?> (not editable here &mdash; disable and recreate the account to change a role).</p>
</header>

<div class="card">
  <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
  <?php if ($saved): ?><div class="flash ok">Saved.</div><?php endif; ?>
  <form method="post" action="admin_user_edit.php?id=<?= (int)$targetId ?>">
    <?= csrf_field() ?>
    <input type="hidden" name="id" value="<?= (int)$targetId ?>">
    <label for="full_name">Full name</label>
    <input type="text" id="full_name" name="full_name" required value="<?= htmlspecialchars($target['full_name'], ENT_QUOTES) ?>">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required value="<?= htmlspecialchars($target['email'], ENT_QUOTES) ?>">
    <label for="phone">Phone</label>
    <input type="text" id="phone" name="phone" value="<?= htmlspecialchars((string)$target['phone'], ENT_QUOTES) ?>">
    <?php if (in_array($target['role'], ['doctor', 'radiologist'], true)): ?>
      <label for="license_number">License number</label>
      <input type="text" id="license_number" name="license_number" value="<?= htmlspecialchars((string)$target['license_number'], ENT_QUOTES) ?>">
    <?php endif; ?>
    <div style="margin-top:20px;"><button type="submit">Save changes</button></div>
  </form>
</div>

<p><a href="admin_users.php">&larr; Back to users</a></p>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
