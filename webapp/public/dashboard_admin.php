<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/inference_client.php';

bootstrap_session();
$user = require_role(['admin']);

$usersByRole = db()->query('SELECT role, COUNT(*) AS c FROM users WHERE is_active = 1 GROUP BY role')->fetchAll(PDO::FETCH_KEY_PAIR);
$scansByStatus = db()->query('SELECT status, COUNT(*) AS c FROM scans GROUP BY status')->fetchAll(PDO::FETCH_KEY_PAIR);
$reviewCount = (int)db()->query("SELECT COUNT(*) AS c FROM reviews WHERE status = 'final'")->fetch()['c'];
$assignmentCount = (int)db()->query('SELECT COUNT(*) AS c FROM doctor_patient WHERE is_active = 1')->fetch()['c'];

$endpointUrl = get_setting('inference_endpoint_url', '');
$health = $endpointUrl !== '' ? check_inference_health() : null;

$pageTitle = 'Admin dashboard';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Overview</h1>
</header>

<div class="card">
  <h2>Inference endpoint</h2>
  <?php if (!$endpointUrl): ?>
    <div class="flash err">Not configured yet. <a href="admin_settings.php">Set it up</a>.</div>
  <?php elseif ($health['reachable']): ?>
    <div class="flash ok">Reachable.</div>
  <?php else: ?>
    <div class="flash err">Unreachable: <?= htmlspecialchars($health['message'], ENT_QUOTES) ?> &mdash; check <a href="admin_settings.php">Settings</a>.</div>
  <?php endif; ?>
</div>

<div class="card">
  <h2>Users</h2>
  <table>
    <thead><tr><th>Role</th><th>Active count</th></tr></thead>
    <tbody>
      <?php foreach (['patient', 'doctor', 'radiologist', 'admin'] as $role): ?>
        <tr><td><?= htmlspecialchars(ucfirst($role), ENT_QUOTES) ?></td><td><?= (int)($usersByRole[$role] ?? 0) ?></td></tr>
      <?php endforeach; ?>
    </tbody>
  </table>
  <p style="margin-top:14px;"><a href="admin_users.php">Manage users</a> &middot; <a href="admin_assign.php">Manage assignments (<?= $assignmentCount ?> active)</a></p>
</div>

<div class="card">
  <h2>Scans</h2>
  <table>
    <thead><tr><th>Status</th><th>Count</th></tr></thead>
    <tbody>
      <?php foreach (['pending', 'processing', 'completed', 'failed'] as $status): ?>
        <tr><td><?= htmlspecialchars(ucfirst($status), ENT_QUOTES) ?></td><td><?= (int)($scansByStatus[$status] ?? 0) ?></td></tr>
      <?php endforeach; ?>
    </tbody>
  </table>
  <p class="lede" style="margin-top:14px;"><?= $reviewCount ?> finalized clinician reviews recorded.</p>
</div>

<div class="card">
  <h2>More</h2>
  <p><a href="admin_audit_log.php">View audit log</a></p>
  <p><a href="admin_settings.php">Inference endpoint settings</a></p>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
