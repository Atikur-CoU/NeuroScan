<?php
/**
 * Include after calling bootstrap_session() and after any header()/redirect
 * logic has already run (this prints output). Optionally set $pageTitle
 * before including.
 */
require_once __DIR__ . '/csrf.php';
$user = current_user();
$title = isset($pageTitle) ? $pageTitle . ' - NeuroScan' : 'NeuroScan';
?>
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title><?= htmlspecialchars($title, ENT_QUOTES) ?></title>
<link rel="stylesheet" href="assets/css/style.css" />
</head>
<body>
<div class="disclaimer-banner">
  Research prototype &mdash; not for clinical use. AI predictions are not a diagnosis; every result must be confirmed by a licensed radiologist or doctor.
</div>
<nav class="topnav">
  <div class="topnav-inner">
    <a class="brand" href="index.php">NeuroScan</a>
    <div class="topnav-links">
      <?php if ($user): ?>
        <span class="pill"><?= htmlspecialchars($user['full_name'], ENT_QUOTES) ?> &middot; <?= htmlspecialchars($user['role'], ENT_QUOTES) ?></span>
        <?php if ($user['role'] === 'patient'): ?>
          <a href="dashboard_patient.php">My Scans</a>
        <?php endif; ?>
        <?php if (in_array($user['role'], ['doctor', 'radiologist'], true)): ?>
          <a href="dashboard_provider.php">Dashboard</a>
        <?php endif; ?>
        <?php if ($user['role'] === 'admin'): ?>
          <a href="dashboard_admin.php">Dashboard</a>
          <a href="admin_users.php">Users</a>
          <a href="admin_assign.php">Assignments</a>
          <a href="admin_audit_log.php">Audit Log</a>
          <a href="admin_settings.php">Settings</a>
        <?php endif; ?>
        <?php if (!empty($user['must_change_password'])): ?>
          <a href="change_password.php">Change password</a>
        <?php endif; ?>
        <form method="post" action="logout.php"><?= csrf_field() ?><button type="submit" class="link-button">Log out</button></form>
      <?php else: ?>
        <a href="login.php">Log in</a>
        <a href="register.php">Register</a>
      <?php endif; ?>
    </div>
  </div>
</nav>
<div class="page<?= isset($narrowPage) && $narrowPage ? ' narrow' : '' ?>">
