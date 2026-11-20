<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';
require_once __DIR__ . '/includes/risk_tier.php';

bootstrap_session();
$user = require_role(['doctor', 'radiologist']);

$uuid = (string)($_GET['scan_uuid'] ?? $_POST['scan_uuid'] ?? '');
if ($uuid === '') {
    http_response_code(404);
    die('Scan not found.');
}

$stmt = db()->prepare(
    "SELECT s.*, p.prediction, p.prob_normal, p.prob_ischemic, p.prob_hemorrhagic, p.heatmap_path, p.likely_not_a_scan, u.full_name AS patient_name
     FROM scans s
     LEFT JOIN predictions p ON p.scan_id = s.id
     JOIN users u ON u.id = s.patient_id
     WHERE s.public_uuid = :uuid"
);
$stmt->execute(['uuid' => $uuid]);
$scan = $stmt->fetch();

if (!$scan) {
    http_response_code(404);
    die('Scan not found.');
}
if (!provider_has_patient_access($user['id'], (int)$scan['patient_id'])) {
    http_response_code(403);
    die('You do not have permission to review this patient.');
}

log_action('scan_viewed', 'scan', (string)$scan['id']);

// The reviewer's most recent review row for this scan, if any.
$stmt = db()->prepare(
    'SELECT * FROM reviews WHERE scan_id = :sid AND reviewer_id = :rid ORDER BY id DESC LIMIT 1'
);
$stmt->execute(['sid' => $scan['id'], 'rid' => $user['id']]);
$myLatest = $stmt->fetch();

$isCorrection = $myLatest && $myLatest['status'] === 'final';
$editable = !$myLatest || $myLatest['status'] === 'draft';

// Other reviewers' finalized reviews on this scan, for context.
$stmt = db()->prepare(
    "SELECT r.*, u.full_name AS reviewer_name
     FROM reviews r JOIN users u ON u.id = r.reviewer_id
     WHERE r.scan_id = :sid AND r.status = 'final'
     ORDER BY r.created_at DESC"
);
$stmt->execute(['sid' => $scan['id']]);
$finalReviews = $stmt->fetchAll();

$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['do'] ?? '') !== '') {
    csrf_check();
    if ($scan['status'] !== 'completed') {
        $error = 'This scan has no completed AI result yet.';
    } else {
        $agrees = $_POST['agrees_with_ai'] ?? '';
        $agreesValue = $agrees === '1' ? 1 : ($agrees === '0' ? 0 : null);
        $override = in_array($_POST['diagnosis_override'] ?? '', ['normal', 'ischemic', 'hemorrhagic', 'other', 'inconclusive'], true)
            ? $_POST['diagnosis_override'] : null;
        $severity = in_array($_POST['severity'] ?? '', ['low', 'moderate', 'high', 'critical'], true)
            ? $_POST['severity'] : null;
        $notes = trim((string)($_POST['notes'] ?? ''));
        $status = ($_POST['do'] === 'finalize') ? 'final' : 'draft';

        if ($editable && $myLatest) {
            $stmt = db()->prepare(
                'UPDATE reviews SET agrees_with_ai = :agrees, diagnosis_override = :override, severity = :severity,
                                     notes = :notes, status = :status
                 WHERE id = :id'
            );
            $stmt->execute([
                'agrees' => $agreesValue, 'override' => $override, 'severity' => $severity,
                'notes' => $notes !== '' ? $notes : null, 'status' => $status, 'id' => $myLatest['id'],
            ]);
            $reviewId = $myLatest['id'];
        } else {
            $amends = $isCorrection ? $myLatest['id'] : null;
            $stmt = db()->prepare(
                'INSERT INTO reviews (scan_id, reviewer_id, reviewer_role, agrees_with_ai, diagnosis_override, severity, notes, status, amends_review_id)
                 VALUES (:sid, :rid, :role, :agrees, :override, :severity, :notes, :status, :amends)'
            );
            $stmt->execute([
                'sid' => $scan['id'], 'rid' => $user['id'], 'role' => $user['role'],
                'agrees' => $agreesValue, 'override' => $override, 'severity' => $severity,
                'notes' => $notes !== '' ? $notes : null, 'status' => $status, 'amends' => $amends,
            ]);
            $reviewId = (int)db()->lastInsertId();
            log_action('review_created', 'review', (string)$reviewId, ['scan_id' => $scan['id']]);
            if ($amends) {
                log_action('review_amended', 'review', (string)$reviewId, ['amends_review_id' => $amends]);
            }
        }

        if ($status === 'final') {
            log_action('review_finalized', 'review', (string)$reviewId, ['scan_id' => $scan['id']]);
            header('Location: dashboard_provider.php');
            exit;
        }
        header('Location: review_form.php?scan_uuid=' . urlencode($uuid));
        exit;
    }
}

// Re-fetch after a draft save so the form reflects the stored state.
if (!$myLatest || $_SERVER['REQUEST_METHOD'] === 'POST') {
    $stmt = db()->prepare(
        'SELECT * FROM reviews WHERE scan_id = :sid AND reviewer_id = :rid ORDER BY id DESC LIMIT 1'
    );
    $stmt->execute(['sid' => $scan['id'], 'rid' => $user['id']]);
    $myLatest = $stmt->fetch();
    $isCorrection = $myLatest && $myLatest['status'] === 'final';
    $editable = !$myLatest || $myLatest['status'] === 'draft';
}

$formValues = $editable && $myLatest ? $myLatest : ['agrees_with_ai' => null, 'diagnosis_override' => null, 'severity' => null, 'notes' => ''];

$pageTitle = 'Review scan';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Review</span>
  <h1><?= htmlspecialchars($scan['patient_name'], ENT_QUOTES) ?> &mdash; <?= htmlspecialchars($scan['modality'], ENT_QUOTES) ?> scan</h1>
  <p class="lede">Uploaded <?= htmlspecialchars($scan['uploaded_at'], ENT_QUOTES) ?></p>
</header>

<div class="preview-row show" style="margin-bottom:20px;">
  <img src="scan_image.php?uuid=<?= urlencode($uuid) ?>" alt="Scan image" />
  <div class="preview-meta">
    <div class="preview-name"><?= htmlspecialchars($scan['original_filename'], ENT_QUOTES) ?></div>
    <div class="preview-sub"><span class="status-badge <?= htmlspecialchars($scan['status'], ENT_QUOTES) ?>"><?= htmlspecialchars($scan['status'], ENT_QUOTES) ?></span></div>
  </div>
</div>

<?php if ($scan['status'] === 'completed' && $scan['prediction']): ?>
  <?php
  $isOod = $scan['likely_not_a_scan'] !== null && (bool)$scan['likely_not_a_scan'];
  $risk = compute_risk_tier($scan['prediction'], (float)$scan['prob_normal'], (float)$scan['prob_ischemic'], (float)$scan['prob_hemorrhagic'], $isOod, (string)$scan['modality']);
  ?>
  <div class="stage" style="margin-bottom:20px;">
    <div class="stage-label">AI Prediction</div>
    <?php if ($isOod): ?>
      <div class="flash err" style="margin-bottom:12px;">
        This image doesn't look like a grayscale CT/MRI brain scan. The result below is likely meaningless &mdash; check the patient's uploaded file.
      </div>
    <?php endif; ?>
    <div class="verdict <?= htmlspecialchars($scan['prediction'], ENT_QUOTES) ?>"><?= htmlspecialchars($scan['prediction'], ENT_QUOTES) ?></div>
    <span style="display:inline-block; margin:6px 0 12px; padding:4px 12px; border-radius:12px; background:<?= htmlspecialchars($risk['color'], ENT_QUOTES) ?>; color:#fff; font-size:0.85em; font-weight:600;"><?= htmlspecialchars($risk['label'], ENT_QUOTES) ?></span>
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
  <?php if ($risk['alert']): ?>
    <script src="assets/js/alert_tone.js"></script>
  <?php endif; ?>
<?php else: ?>
  <div class="card"><p class="lede">This scan has no completed AI result yet &mdash; nothing to review.</p></div>
<?php endif; ?>

<?php if ($finalReviews): ?>
  <div class="card">
    <h2>Existing reviews</h2>
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
<?php endif; ?>

<?php if ($scan['status'] === 'completed'): ?>
  <div class="card">
    <h2><?= $isCorrection ? 'Add a correction' : ($myLatest ? 'Continue your draft' : 'Your review') ?></h2>
    <?php if ($isCorrection): ?>
      <p class="lede">You already finalized a review for this scan. Submitting below records a correction &mdash; your original finalized review is preserved for the audit trail, not overwritten.</p>
    <?php endif; ?>
    <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
    <form method="post" action="review_form.php">
      <?= csrf_field() ?>
      <input type="hidden" name="scan_uuid" value="<?= htmlspecialchars($uuid, ENT_QUOTES) ?>">
      <label for="agrees_with_ai">Do you agree with the AI prediction?</label>
      <select id="agrees_with_ai" name="agrees_with_ai">
        <option value="" <?= $formValues['agrees_with_ai'] === null ? 'selected' : '' ?>>Not specified</option>
        <option value="1" <?= $formValues['agrees_with_ai'] === 1 ? 'selected' : '' ?>>Yes</option>
        <option value="0" <?= $formValues['agrees_with_ai'] === 0 ? 'selected' : '' ?>>No</option>
      </select>
      <label for="diagnosis_override">Your diagnosis (if different, or to confirm)</label>
      <select id="diagnosis_override" name="diagnosis_override">
        <option value="">Not specified</option>
        <?php foreach (['normal', 'ischemic', 'hemorrhagic', 'other', 'inconclusive'] as $opt): ?>
          <option value="<?= $opt ?>" <?= $formValues['diagnosis_override'] === $opt ? 'selected' : '' ?>><?= ucfirst($opt) ?></option>
        <?php endforeach; ?>
      </select>
      <label for="severity">Severity</label>
      <select id="severity" name="severity">
        <option value="">Not specified</option>
        <?php foreach (['low', 'moderate', 'high', 'critical'] as $opt): ?>
          <option value="<?= $opt ?>" <?= $formValues['severity'] === $opt ? 'selected' : '' ?>><?= ucfirst($opt) ?></option>
        <?php endforeach; ?>
      </select>
      <label for="notes">Notes</label>
      <textarea id="notes" name="notes"><?= htmlspecialchars($formValues['notes'] ?? '', ENT_QUOTES) ?></textarea>
      <div style="margin-top:20px; display:flex; gap:10px;">
        <button type="submit" name="do" value="save_draft" class="btn secondary">Save draft</button>
        <button type="submit" name="do" value="finalize">Finalize review</button>
      </div>
    </form>
  </div>
<?php endif; ?>

<p><a href="dashboard_provider.php">&larr; Back to dashboard</a></p>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
