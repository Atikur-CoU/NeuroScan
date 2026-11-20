<?php
/**
 * ONE-TIME, HTTP-reachable equivalent of sql/seed_admin.php, for hosts
 * without SSH access. Guarded by a random token (below) rather than left
 * open, since this is otherwise an unauthenticated admin-account-creation
 * path -- exactly the kind of thing that must never be left live.
 *
 * Usage:
 *   1. Change BOOTSTRAP_TOKEN below to your own long random value before
 *      uploading (don't use the placeholder).
 *   2. Upload this file into public_html/ alongside the rest of public/. It is kept in sql/,
 *      not public/, so that a normal upload of public/ never puts it on the server.
 *   3. Visit https://yourdomain.com/_bootstrap_admin.php?token=<your-token>
 *      once, fill in the form, submit.
 *   4. DELETE THIS FILE from the server immediately after. It refuses to
 *      run at all once an admin account already exists (belt-and-braces),
 *      but don't rely on that -- delete it.
 */

const BOOTSTRAP_TOKEN = 'CHANGE-ME-TO-A-LONG-RANDOM-VALUE-BEFORE-UPLOADING';

require_once __DIR__ . '/includes/db.php';

if (BOOTSTRAP_TOKEN === 'CHANGE-ME-TO-A-LONG-RANDOM-VALUE-BEFORE-UPLOADING') {
    http_response_code(500);
    die('Set BOOTSTRAP_TOKEN to a real random value before using this script.');
}

$token = $_GET['token'] ?? $_POST['token'] ?? '';
if (!hash_equals(BOOTSTRAP_TOKEN, (string)$token)) {
    http_response_code(403);
    die('Forbidden.');
}

$stmt = db()->query("SELECT COUNT(*) AS c FROM users WHERE role = 'admin'");
if ((int)$stmt->fetch()['c'] > 0) {
    die('An admin account already exists. Delete this file.');
}

$error = null;
$success = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = trim((string)($_POST['full_name'] ?? ''));
    $email = trim((string)($_POST['email'] ?? ''));
    $password = (string)($_POST['password'] ?? '');

    if ($name === '' || $email === '' || strlen($password) < 8) {
        $error = 'Fill in all fields; password must be at least 8 characters.';
    } else {
        $stmt = db()->prepare('SELECT id FROM users WHERE email = :email');
        $stmt->execute(['email' => $email]);
        if ($stmt->fetch()) {
            $error = 'A user with that email already exists.';
        } else {
            $stmt = db()->prepare(
                'INSERT INTO users (role, full_name, email, password_hash) VALUES (:role, :name, :email, :hash)'
            );
            $stmt->execute([
                'role' => 'admin',
                'name' => $name,
                'email' => $email,
                'hash' => password_hash($password, PASSWORD_DEFAULT),
            ]);
            $success = 'Admin account created (id ' . db()->lastInsertId() . '). Log in at /login.php, then DELETE THIS FILE now.';
        }
    }
}
?>
<!DOCTYPE html>
<html><body style="font-family:sans-serif;max-width:480px;margin:60px auto;">
<h1>NeuroScan -- create the first admin account</h1>
<?php if ($success): ?>
  <p style="color:green;font-weight:bold;"><?= htmlspecialchars($success, ENT_QUOTES) ?></p>
<?php else: ?>
  <?php if ($error): ?><p style="color:red;"><?= htmlspecialchars($error, ENT_QUOTES) ?></p><?php endif; ?>
  <form method="post">
    <input type="hidden" name="token" value="<?= htmlspecialchars($token, ENT_QUOTES) ?>">
    <p><label>Full name<br><input type="text" name="full_name" style="width:100%;padding:8px;" required></label></p>
    <p><label>Email<br><input type="email" name="email" style="width:100%;padding:8px;" required></label></p>
    <p><label>Password<br><input type="password" name="password" minlength="8" style="width:100%;padding:8px;" required></label></p>
    <button type="submit" style="padding:10px 20px;">Create admin</button>
  </form>
<?php endif; ?>
<p style="color:#900;margin-top:24px;"><strong>Delete this file from the server as soon as you're done.</strong></p>
</body></html>
