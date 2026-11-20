<?php
require_once __DIR__ . '/includes/auth.php';
bootstrap_session();
$user = require_role(['doctor', 'radiologist']);

$stmt = db()->prepare(
    "SELECT s.public_uuid, s.modality, s.uploaded_at, p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic,
            u.full_name AS patient_name
     FROM scans s
     JOIN predictions p ON p.scan_id = s.id
     JOIN users u ON u.id = s.patient_id
     JOIN doctor_patient dp ON dp.patient_id = s.patient_id AND dp.provider_id = :provider_id AND dp.is_active = 1
     LEFT JOIN reviews r ON r.scan_id = s.id AND r.reviewer_id = :provider_id2
     WHERE s.status = 'completed' AND r.id IS NULL
     ORDER BY s.uploaded_at ASC"
);
$stmt->execute(['provider_id' => $user['id'], 'provider_id2' => $user['id']]);
$pending = $stmt->fetchAll();

$stmt = db()->prepare(
    "SELECT u.id, u.full_name, u.email, dp.relationship_type,
            (SELECT COUNT(*) FROM scans WHERE patient_id = u.id) AS scan_count
     FROM doctor_patient dp
     JOIN users u ON u.id = dp.patient_id
     WHERE dp.provider_id = :provider_id AND dp.is_active = 1
     ORDER BY u.full_name"
);
$stmt->execute(['provider_id' => $user['id']]);
$myPatients = $stmt->fetchAll();

$pageTitle = 'Provider dashboard';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; <?= htmlspecialchars(ucfirst($user['role']), ENT_QUOTES) ?></span>
  <h1>Pending my review</h1>
  <p class="lede">Completed AI screenings from your assigned patients that you haven't reviewed yet.</p>
</header>

<div class="card">
  <?php if (!$pending): ?>
    <p class="lede">Nothing pending &mdash; you're caught up.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Patient</th><th>Date</th><th>Modality</th><th>AI result</th><th></th></tr></thead>
      <tbody>
        <?php foreach ($pending as $s): ?>
          <tr>
            <td><?= htmlspecialchars($s['patient_name'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($s['uploaded_at'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($s['modality'], ENT_QUOTES) ?></td>
            <td class="verdict <?= htmlspecialchars($s['prediction'], ENT_QUOTES) ?>" style="font-size:14px;"><?= htmlspecialchars(ucfirst($s['prediction']), ENT_QUOTES) ?></td>
            <td><a href="review_form.php?scan_uuid=<?= urlencode($s['public_uuid']) ?>">Review</a></td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<div class="card">
  <h2>My patients</h2>
  <?php if (!$myPatients): ?>
    <p class="lede">No patients assigned to you yet. An admin manages assignments.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Name</th><th>Email</th><th>Relationship</th><th>Scans</th><th></th></tr></thead>
      <tbody>
        <?php foreach ($myPatients as $p): ?>
          <tr>
            <td><?= htmlspecialchars($p['full_name'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($p['email'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($p['relationship_type'], ENT_QUOTES) ?></td>
            <td><?= (int)$p['scan_count'] ?></td>
            <td><a href="provider_patient.php?patient_id=<?= (int)$p['id'] ?>">View history</a></td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
