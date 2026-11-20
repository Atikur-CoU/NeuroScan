<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';
require_once __DIR__ . '/includes/inference_client.php';
require_once __DIR__ . '/includes/risk_tier.php';
require_once __DIR__ . '/includes/result_gate.php';

bootstrap_session();
$user = require_login();

$uuid = (string)($_GET['uuid'] ?? '');
if ($uuid === '') {
    http_response_code(404);
    die('Scan not found.');
}

$stmt = db()->prepare(
    'SELECT s.*, p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic, p.heatmap_path, p.likely_not_a_scan, p.last_error, p.attempt_count,
            p.pathway, p.evaluated_pathway, p.warnings, p.decision_note, p.model_version
     FROM scans s LEFT JOIN predictions p ON p.scan_id = s.id
     WHERE s.public_uuid = :uuid'
);
$stmt->execute(['uuid' => $uuid]);
$scan = $stmt->fetch();

if (!$scan) {
    http_response_code(404);
    die('Scan not found.');
}

// Ownership / RBAC: a patient may only view their own scan; a doctor or
// radiologist may view it only if actively linked to that patient; admin
// can view any scan (needed for settings/support, always logged below).
$isOwner = $user['role'] === 'patient' && (int)$scan['patient_id'] === (int)$user['id'];
$isProvider = in_array($user['role'], ['doctor', 'radiologist'], true)
    && provider_has_patient_access($user['id'], (int)$scan['patient_id']);
$isAdmin = $user['role'] === 'admin';
if (!$isOwner && !$isProvider && !$isAdmin) {
    http_response_code(403);
    die('You do not have permission to view this scan.');
}

if (!$isOwner) {
    // Any view by someone other than the scan's own patient is logged --
    // this is the "who accessed this record" accountability trail.
    log_action('scan_viewed', 'scan', (string)$scan['id']);
}

$stmt = db()->prepare(
    "SELECT r.*, u.full_name AS reviewer_name
     FROM reviews r JOIN users u ON u.id = r.reviewer_id
     WHERE r.scan_id = :sid AND r.status = 'final'
     ORDER BY r.created_at DESC"
);
$stmt->execute(['sid' => $scan['id']]);
$finalReviews = $stmt->fetchAll();
$resultHidden = result_hidden_from_user($user, (int)$scan['id']);

// Relative, not root-absolute: the live site is deployed under a
// subdirectory (/neuroscan/), so a leading "/" here would resolve to
// the domain root and 404 instead of back to the app.
$backUrl = 'index.php';
if ($isOwner) {
    $backUrl = 'dashboard_patient.php';
} elseif ($isProvider) {
    $backUrl = 'dashboard_provider.php';
} elseif ($isAdmin) {
    $backUrl = 'dashboard_admin.php';
}

$retryError = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'retry') {
    csrf_check();
    if ($user['role'] === 'patient' && (int)$scan['patient_id'] === (int)$user['id']) {
        try {
            run_and_store_prediction($scan);
        } catch (InferenceError $e) {
            $retryError = $e->getMessage();
        }
        header('Location: scan_view2.php?uuid=' . urlencode($uuid));
        exit;
    }
}

$pageTitle = 'Scan';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; <?= htmlspecialchars($scan['modality'], ENT_QUOTES) ?></span>
  <h1>Scan from <?= htmlspecialchars($scan['uploaded_at'], ENT_QUOTES) ?></h1>
</header>

<div class="preview-row show" style="margin-bottom:20px;">
  <img src="scan_image.php?uuid=<?= urlencode($uuid) ?>" alt="Scan image" />
  <div class="preview-meta">
    <div class="preview-name"><?= htmlspecialchars($scan['original_filename'], ENT_QUOTES) ?></div>
    <div class="preview-sub"><?= number_format($scan['file_size_bytes'] / 1024) ?> KB &middot; <span class="status-badge <?= htmlspecialchars($scan['status'], ENT_QUOTES) ?>"><?= htmlspecialchars($scan['status'], ENT_QUOTES) ?></span></div>
  </div>
</div>

<?php if ($scan['notes']): ?>
  <div class="card"><strong>Notes:</strong> <?= nl2br(htmlspecialchars($scan['notes'], ENT_QUOTES)) ?></div>
<?php endif; ?>

<?php if ($resultHidden && $scan['status'] === 'completed'): ?>
  <div class="card"><p class="lede">Your scan has been received. Its result will appear here after a doctor or radiologist has reviewed it. If you are unwell, do not wait: seek medical care now.</p></div>
<?php elseif ($scan['status'] === 'completed' && $scan['prediction']): ?>
  <?php
  $isOod = $scan['likely_not_a_scan'] !== null && (bool)$scan['likely_not_a_scan'];
  $risk = compute_risk_tier($scan['prediction'], (float)$scan['prob_normal'], (float)$scan['prob_ischemic'], (float)$scan['prob_hemorrhagic'], $isOod, (string)$scan['modality']);
  ?>
  <div class="stage">
    <div class="stage-label">AI Prediction</div>
    <?php if ($isOod): ?>
      <div class="flash err" style="margin-bottom:12px;">
        This image doesn't look like a grayscale CT/MRI brain scan. The result below is likely meaningless &mdash; please check you uploaded the correct file.
      </div>
    <?php endif; ?>
    <div class="verdict <?= htmlspecialchars($scan['prediction'], ENT_QUOTES) ?>"><?= htmlspecialchars($scan['prediction'], ENT_QUOTES) ?></div>
    <span style="display:inline-block; margin:6px 0 12px; padding:4px 12px; border-radius:12px; background:<?= htmlspecialchars($risk['color'], ENT_QUOTES) ?>; color:#fff; font-size:0.85em; font-weight:600;"><?= htmlspecialchars($risk['label'], ENT_QUOTES) ?></span>
    <?php $pathwayLabel = pathway_label($scan['pathway']); ?>
    <?php if ($pathwayLabel !== null): ?>
      <div class="preview-sub" style="margin-bottom:8px;">Input: <?= htmlspecialchars($pathwayLabel, ENT_QUOTES) ?></div>
    <?php endif; ?>
    <?php foreach (prediction_warnings($scan['warnings']) as $w): ?>
      <div class="flash" style="margin-bottom:8px;"><?= htmlspecialchars($w, ENT_QUOTES) ?></div>
    <?php endforeach; ?>
    <?php if (!empty($scan['decision_note'])): ?>
      <div class="preview-sub" style="margin-bottom:8px;"><?= htmlspecialchars($scan['decision_note'], ENT_QUOTES) ?></div>
    <?php endif; ?>
    <div class="prob-list">
      <?php
      $rows = [
          ['normal', (float)$scan['prob_normal']],
          ['ischemic', (float)$scan['prob_ischemic']],
          ['hemorrhagic', (float)$scan['prob_hemorrhagic']],
      ];
      usort($rows, fn($a, $b) => $b[1] <=> $a[1]);
      foreach ($rows as [$label, $prob]):
      ?>
        <div class="prob-row">
          <span class="prob-row-label <?= $label ?>"><?= $label ?></span>
          <div class="prob-track"><div class="prob-fill <?= $label ?>" style="width:<?= round($prob * 100, 1) ?>%"></div></div>
          <span class="prob-pct"><?= number_format($prob * 100, 1) ?>%</span>
        </div>
      <?php endforeach; ?>
    </div>
    <?php if ($scan['heatmap_path']): ?>
      <div style="margin-top:16px;">
        <img src="scan_image.php?uuid=<?= urlencode($uuid) ?>&type=heatmap" alt="Grad-CAM heatmap" style="max-width:100%; border-radius:8px;" />
        <div class="preview-sub" style="margin-top:6px;">Regions where the model's evidence is concentrated (Grad-CAM activation regions) &mdash; not a lesion boundary. Always confirm with a licensed radiologist or doctor.</div>
      </div>
    <?php endif; ?>
  </div>
  <?php if ($risk['alert'] && !($isOwner && result_gate_enabled())): ?>
    <script src="assets/js/alert_tone.js"></script>
  <?php endif; ?>
<?php elseif ($scan['status'] === 'failed'): ?>
  <div class="card">
    <div class="flash err">
      Prediction service temporarily unavailable<?= $scan['last_error'] ? ' (' . htmlspecialchars($scan['last_error'], ENT_QUOTES) . ')' : '' ?>. This can happen if the notebook running the AI model isn't active right now.
    </div>
    <?php if ($user['role'] === 'patient'): ?>
      <form method="post" action="scan_view2.php?uuid=<?= urlencode($uuid) ?>">
        <?= csrf_field() ?>
        <input type="hidden" name="action" value="retry">
        <button type="submit">Retry prediction</button>
      </form>
    <?php endif; ?>
  </div>
<?php else: ?>
  <div class="card"><p class="lede">Prediction in progress&hellip;</p></div>
<?php endif; ?>

<?php if ($finalReviews): ?>
  <div class="card">
    <h2>Clinician review</h2>
    <?php foreach ($finalReviews as $r): ?>
      <div style="padding:10px 0; border-bottom:1px solid var(--border);">
        <strong><?= htmlspecialchars($r['reviewer_name'], ENT_QUOTES) ?></strong>
        (<?= htmlspecialchars($r['reviewer_role'], ENT_QUOTES) ?>) &middot; <?= htmlspecialchars($r['created_at'], ENT_QUOTES) ?><br>
        <?php if ($r['agrees_with_ai'] !== null): ?>Agrees with AI: <?= $r['agrees_with_ai'] ? 'Yes' : 'No' ?><br><?php endif; ?>
        <?php if ($r['diagnosis_override']): ?>Diagnosis: <?= htmlspecialchars($r['diagnosis_override'], ENT_QUOTES) ?><br><?php endif; ?>
        <?php if ($r['severity']): ?>Severity: <?= htmlspecialchars($r['severity'], ENT_QUOTES) ?><br><?php endif; ?>
        <?php if ($r['notes']): ?><?= nl2br(htmlspecialchars($r['notes'], ENT_QUOTES)) ?><?php endif; ?>
      </div>
    <?php endforeach; ?>
  </div>
<?php elseif ($isOwner && !$resultHidden && $scan['status'] === 'completed'): ?>
  <div class="card"><p class="lede">No clinician has reviewed this scan yet.</p></div>
<?php endif; ?>

<?php if ($isProvider && $scan['status'] === 'completed'): ?>
  <p><a href="review_form.php?scan_uuid=<?= urlencode($uuid) ?>">Write a review &rarr;</a></p>
<?php endif; ?>

<p style="margin-top:20px;"><a href="<?= htmlspecialchars($backUrl, ENT_QUOTES) ?>">&larr; Back</a></p>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
