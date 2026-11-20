<?php
require_once __DIR__ . '/includes/auth.php';
bootstrap_session();
$user = require_role(['doctor', 'radiologist']);

$patientId = (int)($_GET['patient_id'] ?? 0);
if (!$patientId || !provider_has_patient_access($user['id'], $patientId)) {
    http_response_code(403);
    die('You do not have permission to view this patient.');
}

$stmt = db()->prepare('SELECT full_name, email FROM users WHERE id = :id');
$stmt->execute(['id' => $patientId]);
$patient = $stmt->fetch();

$stmt = db()->prepare(
    "SELECT s.public_uuid, s.modality, s.status, s.uploaded_at, p.prediction,
            (SELECT COUNT(*) FROM reviews WHERE scan_id = s.id AND status = 'final') AS review_count
     FROM scans s
     LEFT JOIN predictions p ON p.scan_id = s.id
     WHERE s.patient_id = :pid
     ORDER BY s.uploaded_at DESC"
);
$stmt->execute(['pid' => $patientId]);
$scans = $stmt->fetchAll();

$pageTitle = 'Patient history';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; <?= htmlspecialchars(ucfirst($user['role']), ENT_QUOTES) ?></span>
  <h1><?= htmlspecialchars($patient['full_name'], ENT_QUOTES) ?></h1>
  <p class="lede"><?= htmlspecialchars($patient['email'], ENT_QUOTES) ?></p>
</header>

<div class="card">
  <?php if (!$scans): ?>
    <p class="lede">No scans yet.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Date</th><th>Modality</th><th>Status</th><th>AI result</th><th>Reviewed</th><th></th></tr></thead>
      <tbody>
        <?php foreach ($scans as $s): ?>
          <tr>
            <td><?= htmlspecialchars($s['uploaded_at'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($s['modality'], ENT_QUOTES) ?></td>
            <td><span class="status-badge <?= htmlspecialchars($s['status'], ENT_QUOTES) ?>"><?= htmlspecialchars($s['status'], ENT_QUOTES) ?></span></td>
            <td><?= $s['prediction'] ? htmlspecialchars(ucfirst($s['prediction']), ENT_QUOTES) : '&mdash;' ?></td>
            <td><?= $s['review_count'] > 0 ? 'Yes' : 'No' ?></td>
            <td>
              <a href="scan_view2.php?uuid=<?= urlencode($s['public_uuid']) ?>">View</a>
              <?php if ($s['status'] === 'completed'): ?>
                &middot; <a href="review_form.php?scan_uuid=<?= urlencode($s['public_uuid']) ?>">Review</a>
              <?php endif; ?>
            </td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<p><a href="dashboard_provider.php">&larr; Back to dashboard</a></p>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
