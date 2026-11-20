<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/result_gate.php';
bootstrap_session();
$user = require_role(['patient']);

$stmt = db()->prepare(
    "SELECT s.public_uuid, s.modality, s.status, s.uploaded_at, p.prediction,
            (SELECT COUNT(*) FROM reviews r WHERE r.scan_id = s.id AND r.status = 'final') AS final_review_count
     FROM scans s
     LEFT JOIN predictions p ON p.scan_id = s.id
     WHERE s.patient_id = :pid
     ORDER BY s.uploaded_at DESC"
);
$stmt->execute(['pid' => $user['id']]);
$scans = $stmt->fetchAll();
$gate = result_gate_enabled();

$pageTitle = 'My Scans';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan</span>
  <h1>My scans</h1>
  <p class="lede"><?= $gate
      ? 'Upload a CT or MRI slice for AI-assisted screening. Results appear after a doctor or radiologist has reviewed them.'
      : 'Upload a CT or MRI slice to get an AI-assisted screening result. A clinician review may take longer to appear than the AI result.' ?></p>
</header>

<div class="card">
  <a class="btn" href="scan_upload.php">Upload a new scan</a>
</div>

<div class="card">
  <h2>History</h2>
  <?php if (!$scans): ?>
    <p class="lede">No scans uploaded yet.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Date</th><th>Modality</th><th>Status</th><th>AI result</th><th></th></tr></thead>
      <tbody>
        <?php foreach ($scans as $s): ?>
          <tr>
            <td><?= htmlspecialchars($s['uploaded_at'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($s['modality'], ENT_QUOTES) ?></td>
            <td><span class="status-badge <?= htmlspecialchars($s['status'], ENT_QUOTES) ?>"><?= htmlspecialchars($s['status'], ENT_QUOTES) ?></span></td>
            <?php if ($gate && (int)$s['final_review_count'] === 0): ?>
              <td><?= $s['status'] === 'completed' ? 'Awaiting clinician review' : '&mdash;' ?></td>
            <?php else: ?>
              <td><?= $s['prediction'] ? htmlspecialchars(ucfirst($s['prediction']), ENT_QUOTES) : '&mdash;' ?></td>
            <?php endif; ?>
            <td><a href="scan_view2.php?uuid=<?= urlencode($s['public_uuid']) ?>">View</a></td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
