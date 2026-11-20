<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';
require_once __DIR__ . '/includes/upload_validate.php';
require_once __DIR__ . '/includes/inference_client.php';
require_once __DIR__ . '/includes/result_gate.php';

bootstrap_session();
$user = require_role(['patient']);

$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    csrf_check();
    $modality = ($_POST['modality'] ?? '') === 'MRI' ? 'MRI' : 'CT';
    $notes = trim((string)($_POST['notes'] ?? ''));
    $maxMb = (int)get_setting('max_upload_mb', '10');

    try {
        if (empty($_FILES['file'])) {
            throw new UploadValidationError('Please choose an image file.');
        }
        $stored = validate_and_store_upload($_FILES['file'], $maxMb * 1024 * 1024);
        $context = store_context_uploads($_FILES, $modality, $maxMb * 1024 * 1024);

        $uuid = sprintf(
            '%08x-%04x-%04x-%04x-%012x',
            random_int(0, 0xffffffff),
            random_int(0, 0xffff),
            random_int(0, 0xffff) & 0x0fff | 0x4000,
            random_int(0, 0xffff) & 0x3fff | 0x8000,
            random_int(0, 0xffffffffffff)
        );

        db()->beginTransaction();
        $stmt = db()->prepare(
            'INSERT INTO scans (public_uuid, patient_id, uploaded_by, modality, original_filename, stored_filename, storage_path, context_files, file_size_bytes, mime_type, sha256_hash, status, notes)
             VALUES (:uuid, :pid, :uid, :modality, :orig, :stored, :path, :context, :size, :mime, :hash, :status, :notes)'
        );
        $stmt->execute([
            'uuid' => $uuid,
            'pid' => $user['id'],
            'uid' => $user['id'],
            'modality' => $modality,
            'orig' => $_FILES['file']['name'],
            'stored' => $stored['stored_filename'],
            'path' => $stored['storage_path'],
            'context' => $context ? json_encode($context) : null,
            'size' => $stored['file_size_bytes'],
            'mime' => $stored['mime_type'],
            'hash' => $stored['sha256_hash'],
            'status' => 'processing',
            'notes' => $notes !== '' ? $notes : null,
        ]);
        $scanId = (int)db()->lastInsertId();

        $stmt = db()->prepare('INSERT INTO predictions (scan_id, request_started_at) VALUES (:sid, UTC_TIMESTAMP())');
        $stmt->execute(['sid' => $scanId]);
        db()->commit();

        log_action('scan_uploaded', 'scan', (string)$scanId);

        $scanRow = db()->prepare('SELECT * FROM scans WHERE id = :sid');
        $scanRow->execute(['sid' => $scanId]);
        try {
            run_and_store_prediction($scanRow->fetch());
        } catch (InferenceError $e) {
            // Recorded on the scan by run_and_store_prediction; the scan page shows it with a retry button.
        }

        header('Location: scan_view2.php?uuid=' . urlencode($uuid));
        exit;
    } catch (UploadValidationError $e) {
        $error = $e->getMessage();
    }
}

$pageTitle = 'Upload scan';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan</span>
  <h1>Upload a scan</h1>
</header>

<div class="card">
  <?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>
  <form method="post" action="scan_upload.php" enctype="multipart/form-data">
    <?= csrf_field() ?>
    <label for="modality">Modality</label>
    <select id="modality" name="modality">
      <option value="CT">CT</option>
      <option value="MRI">MRI</option>
    </select>
    <label for="notes">Notes (optional)</label>
    <textarea id="notes" name="notes" placeholder="e.g. symptom onset time, relevant history"></textarea>
    <label for="file-input" style="margin-top:18px;" id="main-label">Image (CT: the slice to assess)</label>
    <div class="dropzone" id="dropzone">
      <input type="file" id="file-input" name="file" accept="image/png,image/jpeg" required />
      <div class="dz-title">Drop a scan here, or click to browse</div>
      <div class="dz-sub">PNG or JPG</div>
    </div>
    <div class="preview-row" id="preview-row">
      <img id="preview-img" alt="Preview" />
      <div class="preview-meta">
        <div class="preview-name" id="preview-name"></div>
        <div class="preview-sub" id="preview-sub"></div>
      </div>
    </div>
    <fieldset id="ctx-ct" style="margin-top:18px;">
      <legend>Adjacent slices (recommended)</legend>
      <p class="preview-sub">The model was evaluated with the slice just above and just below the main slice. Upload both, or neither; with the main slice alone the result comes from a pathway that was not evaluated.</p>
      <label for="prev-file">Previous slice</label>
      <input type="file" id="prev-file" name="prev_file" accept="image/png,image/jpeg" />
      <label for="next-file">Next slice</label>
      <input type="file" id="next-file" name="next_file" accept="image/png,image/jpeg" />
    </fieldset>
    <fieldset id="ctx-mri" style="margin-top:18px; display:none;">
      <legend>ADC image (required for MRI)</legend>
      <p class="preview-sub">Upload the DWI slice above and the ADC map of the same slice here. The model needs both: with one sequence alone it does not detect ischemia. MRI results cover normal versus ischemic only.</p>
      <label for="adc-file">ADC image</label>
      <input type="file" id="adc-file" name="adc_file" accept="image/png,image/jpeg" />
    </fieldset>
    <div style="margin-top:20px;"><button type="submit"><?= result_gate_enabled() ? 'Upload for clinician review' : 'Upload &amp; run AI screening' ?></button></div>
  </form>
</div>

<script src="assets/js/scan_upload.js"></script>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
