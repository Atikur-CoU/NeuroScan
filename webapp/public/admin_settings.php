<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';
require_once __DIR__ . '/includes/inference_client.php';
require_once __DIR__ . '/includes/result_gate.php';

bootstrap_session();
$user = require_role(['admin']);

$saved = false;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'save') {
    csrf_check();
    $before = [
        'inference_endpoint_url' => get_setting('inference_endpoint_url', ''),
        'inference_timeout_seconds' => get_setting('inference_timeout_seconds', ''),
        'inference_connect_timeout_seconds' => get_setting('inference_connect_timeout_seconds', ''),
        'max_upload_mb' => get_setting('max_upload_mb', ''),
        'gate_patient_results' => get_setting('gate_patient_results', '0'),
    ];
    set_setting('inference_endpoint_url', trim((string)($_POST['inference_endpoint_url'] ?? '')), $user['id']);
    $apiKey = (string)($_POST['inference_api_key'] ?? '');
    if ($apiKey !== '') {
        // Only overwrite the key if a new one was actually typed -- the
        // field is left blank on render so the existing secret never
        // round-trips into the page source.
        set_setting('inference_api_key', $apiKey, $user['id']);
    }
    set_setting('inference_timeout_seconds', (string)max(1, (int)($_POST['inference_timeout_seconds'] ?? 30)), $user['id']);
    set_setting('inference_connect_timeout_seconds', (string)max(1, (int)($_POST['inference_connect_timeout_seconds'] ?? 8)), $user['id']);
    set_setting('max_upload_mb', (string)max(1, (int)($_POST['max_upload_mb'] ?? 10)), $user['id']);
    set_setting('gate_patient_results', isset($_POST['gate_patient_results']) ? '1' : '0', $user['id']);
    log_action('settings_updated', 'settings', null, ['before' => $before]);
    $saved = true;
}

$health = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'test') {
    csrf_check();
    $health = check_inference_health();
}

$endpointUrl = get_setting('inference_endpoint_url', '');
$timeout = get_setting('inference_timeout_seconds', '30');
$connectTimeout = get_setting('inference_connect_timeout_seconds', '8');
$maxUploadMb = get_setting('max_upload_mb', '10');

$pageTitle = 'Settings';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Inference endpoint settings</h1>
  <p class="lede">The AI model runs on an always-on Render deployment (see <code>render_inference_server/</code>). Its URL is permanent and shouldn't need updating in normal operation &mdash; this page is here for the rare case the endpoint moves.</p>
</header>

<?php if ($saved): ?><div class="flash ok">Settings saved.</div><?php endif; ?>
<?php if ($health): ?>
  <div class="flash <?= $health['reachable'] ? 'ok' : 'err' ?>">
    <?= $health['reachable'] ? 'Endpoint reachable.' : 'Endpoint unreachable: ' . htmlspecialchars($health['message'], ENT_QUOTES) ?>
    <?php if (!empty($health['data']['checkpoint'])): ?>
      &mdash; checkpoint epoch <?= (int)$health['data']['checkpoint']['epoch'] ?>, val_acc <?= number_format((float)$health['data']['checkpoint']['val_acc'] * 100, 1) ?>%
    <?php endif; ?>
  </div>
<?php endif; ?>

<div class="card">
  <form method="post" action="admin_settings.php">
    <?= csrf_field() ?>
    <input type="hidden" name="action" value="save">
    <label for="inference_endpoint_url">Endpoint base URL</label>
    <input type="text" id="inference_endpoint_url" name="inference_endpoint_url" placeholder="https://xxxx.ngrok-free.app" value="<?= htmlspecialchars($endpointUrl, ENT_QUOTES) ?>">
    <label for="inference_api_key">API key</label>
    <input type="text" id="inference_api_key" name="inference_api_key" placeholder="leave blank to keep the current key">
    <div class="field-hint">Must match the key configured in the notebook's <code>X-API-Key</code> check.</div>
    <label for="inference_timeout_seconds">Request timeout (seconds)</label>
    <input type="text" id="inference_timeout_seconds" name="inference_timeout_seconds" value="<?= htmlspecialchars($timeout, ENT_QUOTES) ?>">
    <label for="inference_connect_timeout_seconds">Connect timeout (seconds)</label>
    <input type="text" id="inference_connect_timeout_seconds" name="inference_connect_timeout_seconds" value="<?= htmlspecialchars($connectTimeout, ENT_QUOTES) ?>">
    <label for="max_upload_mb">Max upload size (MB)</label>
    <input type="text" id="max_upload_mb" name="max_upload_mb" value="<?= htmlspecialchars($maxUploadMb, ENT_QUOTES) ?>">
    <label style="margin-top:18px;"><input type="checkbox" name="gate_patient_results" value="1" <?= result_gate_enabled() ? 'checked' : '' ?>> Hide AI results from patients until a clinician has finalized a review</label>
    <div class="field-hint">Patients without an assigned doctor or radiologist will not see a result until one is assigned and reviews the scan.</div>
    <div style="margin-top:20px;"><button type="submit">Save settings</button></div>
  </form>
</div>

<div class="card">
  <form method="post" action="admin_settings.php">
    <?= csrf_field() ?>
    <input type="hidden" name="action" value="test">
    <button type="submit" class="btn secondary">Test connection</button>
  </form>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
