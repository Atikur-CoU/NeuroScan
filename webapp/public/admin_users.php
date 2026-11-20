<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();
$user = require_role(['admin']);

$error = null;
$createdCredentials = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'create_clinician') {
    csrf_check();
    $fullName = trim((string)($_POST['full_name'] ?? ''));
    $email = trim((string)($_POST['email'] ?? ''));
    $role = ($_POST['role'] ?? '') === 'radiologist' ? 'radiologist' : 'doctor';
    $licenseNumber = trim((string)($_POST['license_number'] ?? ''));

    if ($fullName === '' || $email === '') {
        $error = 'Please fill in name and email.';
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error = 'Please enter a valid email address.';
    } else {
        $stmt = db()->prepare('SELECT id FROM users WHERE email = :email');
        $stmt->execute(['email' => $email]);
        if ($stmt->fetch()) {
            $error = 'An account with that email already exists.';
        } else {
            $tempPassword = bin2hex(random_bytes(6));
            $stmt = db()->prepare(
                'INSERT INTO users (role, full_name, email, password_hash, license_number, must_change_password, created_by)
                 VALUES (:role, :name, :email, :hash, :license, 1, :created_by)'
            );
            $stmt->execute([
                'role' => $role,
                'name' => $fullName,
                'email' => $email,
                'hash' => password_hash($tempPassword, PASSWORD_DEFAULT),
                'license' => $licenseNumber !== '' ? $licenseNumber : null,
                'created_by' => $user['id'],
            ]);
            $newId = (int)db()->lastInsertId();
            log_action('user_created', 'user', (string)$newId, ['role' => $role, 'admin_provisioned' => true]);
            $createdCredentials = ['email' => $email, 'password' => $tempPassword];
        }
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'toggle_active') {
    csrf_check();
    $targetId = (int)($_POST['user_id'] ?? 0);
    if ($targetId && $targetId !== $user['id']) {
        $stmt = db()->prepare('SELECT is_active FROM users WHERE id = :id');
        $stmt->execute(['id' => $targetId]);
        $row = $stmt->fetch();
        if ($row) {
            $newActive = $row['is_active'] ? 0 : 1;
            db()->prepare('UPDATE users SET is_active = :active WHERE id = :id')
                ->execute(['active' => $newActive, 'id' => $targetId]);
            log_action($newActive ? 'user_updated' : 'user_deactivated', 'user', (string)$targetId);
        }
    }
    header('Location: admin_users.php');
    exit;
}

$users = db()->query(
    'SELECT id, role, full_name, email, license_number, is_active, created_at FROM users ORDER BY role, full_name'
)->fetchAll();

$pageTitle = 'Users';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Users</h1>
  <p class="lede">Patients self-register. Doctor and radiologist accounts must be created here.</p>
</header>

<?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
<?php if ($createdCredentials): ?>
  <div class="flash ok">
    Account created for <?= htmlspecialchars($createdCredentials['email'], ENT_QUOTES) ?>.
    Temporary password: <code><?= htmlspecialchars($createdCredentials['password'], ENT_QUOTES) ?></code>
    &mdash; share this with them directly (not by email); they'll be required to set their own password on first login.
  </div>
<?php endif; ?>

<div class="card">
  <h2>Create doctor / radiologist account</h2>
  <form method="post" action="admin_users.php">
    <?= csrf_field() ?>
    <input type="hidden" name="action" value="create_clinician">
    <label for="full_name">Full name</label>
    <input type="text" id="full_name" name="full_name" required>
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required>
    <label for="role">Role</label>
    <select id="role" name="role">
      <option value="doctor">Doctor</option>
      <option value="radiologist">Radiologist</option>
    </select>
    <label for="license_number">License number (optional)</label>
    <input type="text" id="license_number" name="license_number">
    <div style="margin-top:20px;"><button type="submit">Create account</button></div>
  </form>
</div>

<div class="card">
  <h2>All users</h2>
  <table>
    <thead><tr><th>Name</th><th>Role</th><th>Email</th><th>Status</th><th>Joined</th><th></th></tr></thead>
    <tbody>
      <?php foreach ($users as $u): ?>
        <tr>
          <td><?= htmlspecialchars($u['full_name'], ENT_QUOTES) ?></td>
          <td><?= htmlspecialchars($u['role'], ENT_QUOTES) ?></td>
          <td><?= htmlspecialchars($u['email'], ENT_QUOTES) ?></td>
          <td><span class="status-badge <?= $u['is_active'] ? 'completed' : 'failed' ?>"><?= $u['is_active'] ? 'active' : 'disabled' ?></span></td>
          <td><?= htmlspecialchars($u['created_at'], ENT_QUOTES) ?></td>
          <td>
            <a href="admin_user_edit.php?id=<?= (int)$u['id'] ?>">Edit</a>
            <?php if ((int)$u['id'] !== $user['id']): ?>
              <form method="post" action="admin_users.php" style="display:inline;">
                <?= csrf_field() ?>
                <input type="hidden" name="action" value="toggle_active">
                <input type="hidden" name="user_id" value="<?= (int)$u['id'] ?>">
                <button type="submit" class="btn secondary" style="padding:4px 10px; font-size:12px;"><?= $u['is_active'] ? 'Disable' : 'Enable' ?></button>
              </form>
            <?php endif; ?>
          </td>
        </tr>
      <?php endforeach; ?>
    </tbody>
  </table>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
