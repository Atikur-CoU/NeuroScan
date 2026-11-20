<?php
require_once __DIR__ . '/db.php';
require_once __DIR__ . '/audit.php';
require_once __DIR__ . '/upload_validate.php';

class InferenceError extends Exception {}

function get_setting(string $key, ?string $default = null): ?string
{
    $stmt = db()->prepare('SELECT setting_value FROM settings WHERE setting_key = :key');
    $stmt->execute(['key' => $key]);
    $row = $stmt->fetch();
    return $row ? $row['setting_value'] : $default;
}

function set_setting(string $key, string $value, ?int $updatedBy): void
{
    $stmt = db()->prepare(
        'UPDATE settings SET setting_value = :value, updated_by = :by WHERE setting_key = :key'
    );
    $stmt->execute(['value' => $value, 'by' => $updatedBy, 'key' => $key]);
}

/**
 * Calls the configured inference endpoint's /api/predict with the given
 * image file, the scan's modality and any stored context images
 * ($context: endpoint field name => full path, e.g. prev_file / next_file
 * for CT adjacent slices, adc_file for MRI). Returns the decoded, validated
 * response array on success. Throws InferenceError on any failure --
 * callers must catch this and mark the scan as failed rather than letting
 * it propagate as a 500.
 */
function call_inference_endpoint(string $imagePath, string $displayName, string $modality = 'CT', array $context = []): array
{
    $endpoint = rtrim((string)get_setting('inference_endpoint_url', ''), '/');
    if ($endpoint === '') {
        throw new InferenceError('No inference endpoint is configured yet. An admin must set one in Settings.');
    }
    $apiKey = (string)get_setting('inference_api_key', '');
    $timeout = (int)get_setting('inference_timeout_seconds', '30');
    $connectTimeout = (int)get_setting('inference_connect_timeout_seconds', '8');

    $response = http_post_image($endpoint . '/api/predict', $imagePath, $displayName, $apiKey, $timeout, $connectTimeout,
        $modality, $context);
    $data = json_decode($response, true);

    if (!is_array($data)
        || !isset($data['prediction'], $data['prob_normal'], $data['prob_ischemic'], $data['prob_hemorrhagic'])
    ) {
        throw new InferenceError('Inference endpoint returned an unexpected response.');
    }
    if (!in_array($data['prediction'], ['normal', 'ischemic', 'hemorrhagic'], true)) {
        throw new InferenceError('Inference endpoint returned an unknown prediction label.');
    }
    foreach (['prob_normal', 'prob_ischemic', 'prob_hemorrhagic'] as $k) {
        $v = $data[$k];
        if (!is_numeric($v) || $v < -0.0001 || $v > 1.0001) {
            throw new InferenceError('Inference endpoint returned an out-of-range probability.');
        }
    }

    $data['_endpoint'] = $endpoint;
    return $data;
}

/**
 * Base64-decodes and saves the Grad-CAM heatmap from an inference result,
 * if the endpoint included one -- older or heatmap-less endpoints simply
 * won't have this key, and that's not an error, just nothing to save.
 * Returns the relative storage path to record in predictions.heatmap_path,
 * or null if there was nothing to save.
 */
function save_heatmap_if_present(array $result, string $scanStoredFilename): ?string
{
    if (empty($result['heatmap_png_base64'])) {
        return null;
    }
    $bytes = base64_decode($result['heatmap_png_base64'], true);
    if ($bytes === false) {
        return null;
    }
    // $scanStoredFilename already ends in .png -- every upload is
    // re-encoded to PNG server-side regardless of original format
    // (see upload_validate.php), so no extension needs adding here.
    $relativePath = 'heatmap_' . $scanStoredFilename;
    $fullPath = UPLOADS_DIR . '/' . $relativePath;
    if (file_put_contents($fullPath, $bytes) === false) {
        return null;
    }
    return $relativePath;
}

/**
 * Normalizes the inference result's out-of-distribution flag for storage.
 * Older or guard-less endpoints simply won't have this key -- that's not
 * an error, just "unknown," stored as NULL rather than assumed false.
 */
function likely_not_a_scan_value(array $result): ?int
{
    if (!array_key_exists('likely_not_a_scan', $result) || $result['likely_not_a_scan'] === null) {
        return null;
    }
    return $result['likely_not_a_scan'] ? 1 : 0;
}

function check_inference_health(): array
{
    $endpoint = rtrim((string)get_setting('inference_endpoint_url', ''), '/');
    if ($endpoint === '') {
        return ['reachable' => false, 'message' => 'No endpoint configured.'];
    }
    $apiKey = (string)get_setting('inference_api_key', '');

    $ch = curl_init($endpoint . '/api/health');
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 10,
        CURLOPT_CONNECTTIMEOUT => 8,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_HTTPHEADER => [
            'X-API-Key: ' . $apiKey,
            'ngrok-skip-browser-warning: true',
        ],
    ]);
    $body = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err = curl_error($ch);
    curl_close($ch);

    if ($body === false || $httpCode !== 200) {
        return ['reachable' => false, 'message' => $err ?: ('HTTP ' . $httpCode)];
    }
    $data = json_decode($body, true);
    return ['reachable' => true, 'message' => 'OK', 'data' => $data];
}

function http_post_image(string $url, string $filePath, string $displayName, string $apiKey, int $timeout, int $connectTimeout,
                         string $modality = 'CT', array $context = []): string
{
    $ch = curl_init($url);
    $fields = ['file' => new CURLFile($filePath, 'image/png', $displayName), 'modality' => $modality];
    foreach ($context as $field => $path) {
        $fields[$field] = new CURLFile($path, 'image/png', $field . '.png');
    }
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $fields,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => $timeout,
        CURLOPT_CONNECTTIMEOUT => $connectTimeout,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_HTTPHEADER => [
            'X-API-Key: ' . $apiKey,
            'ngrok-skip-browser-warning: true',
        ],
    ]);
    $body = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err = curl_error($ch);
    curl_close($ch);

    if ($body === false) {
        throw new InferenceError('Could not reach the inference endpoint: ' . $err);
    }
    if ($httpCode !== 200) {
        throw new InferenceError('Inference endpoint returned HTTP ' . $httpCode . '.');
    }
    return $body;
}


// ---------------------------------------------------------------------------
// v2 inference server: context images, input pathway, and storing a result.
// ---------------------------------------------------------------------------

/** Form/API field name => key in scans.context_files, per modality. */
const CONTEXT_FIELDS = [
    'CT' => ['prev_file' => 'prev', 'next_file' => 'next'],
    'MRI' => ['adc_file' => 'adc'],
];

/**
 * Validates and stores the context images sent with an upload (CT:
 * previous + next slice, optional, both or neither; MRI: ADC, required). Returns the
 * context map for scans.context_files (empty when none were sent). Files
 * for the other modality are rejected rather than silently ignored.
 */
function store_context_uploads(array $files, string $modality, int $maxBytes): array
{
    $present = static fn(string $f): bool => !empty($files[$f]) && ($files[$f]['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_NO_FILE;
    foreach (CONTEXT_FIELDS as $mod => $fields) {
        if ($mod !== $modality) {
            foreach (array_keys($fields) as $f) {
                if ($present($f)) {
                    throw new UploadValidationError($modality === 'CT'
                        ? 'An ADC image applies to MRI only.'
                        : 'Previous/next slices apply to CT only; for MRI upload the ADC image.');
                }
            }
        }
    }
    if ($modality === 'CT' && $present('prev_file') !== $present('next_file')) {
        throw new UploadValidationError('For CT, upload both the previous and the next slice (or neither).');
    }
    // One MRI sequence alone is refused: the model detected almost no ischemia from DWI or ADC alone (reviewer
    // experiment C2), so the inference server rejects it too.
    if ($modality === 'MRI' && !$present('adc_file')) {
        throw new UploadValidationError('For MRI, upload both the DWI image and the ADC map of the same slice.');
    }
    $context = [];
    foreach (CONTEXT_FIELDS[$modality] as $field => $key) {
        if ($present($field)) {
            $context[$key] = validate_and_store_upload($files[$field], $maxBytes)['storage_path'];
        }
    }
    return $context;
}

/** scans.context_files (JSON) => endpoint field name => full path. */
function context_paths_for_scan(array $scan): array
{
    $stored = json_decode((string)($scan['context_files'] ?? ''), true);
    if (!is_array($stored)) {
        return [];
    }
    $out = [];
    foreach (CONTEXT_FIELDS[$scan['modality']] ?? [] as $field => $key) {
        if (!empty($stored[$key])) {
            $out[$field] = UPLOADS_DIR . '/' . basename((string)$stored[$key]);
        }
    }
    return $out;
}

/**
 * Runs the prediction for a stored scan row and records the outcome
 * (predictions + scans.status + audit log). Shared by the web upload, the
 * API upload and the retry button so the three stay identical. Returns the
 * endpoint result; rethrows InferenceError after marking the scan failed.
 */
function run_and_store_prediction(array $scan, array $auditExtra = []): array
{
    $scanId = (int)$scan['id'];
    log_action('prediction_requested', 'scan', (string)$scanId, $auditExtra);
    try {
        $result = call_inference_endpoint(
            UPLOADS_DIR . '/' . $scan['storage_path'],
            $scan['original_filename'],
            $scan['modality'],
            context_paths_for_scan($scan)
        );
        $heatmapPath = save_heatmap_if_present($result, $scan['stored_filename']);
        $warnings = isset($result['warnings']) && is_array($result['warnings']) ? $result['warnings'] : null;
        $stmt = db()->prepare(
            'UPDATE predictions
             SET prediction = :pred, prob_normal = :pn, prob_ischemic = :pi, prob_hemorrhagic = :ph,
                 heatmap_path = :heatmap, likely_not_a_scan = :ood, inference_endpoint = :endpoint,
                 model_version = :mv, pathway = :pathway, evaluated_pathway = :evaluated, warnings = :warnings,
                 decision_note = :note, attempt_count = attempt_count + 1, last_error = NULL,
                 request_completed_at = UTC_TIMESTAMP()
             WHERE scan_id = :sid'
        );
        $stmt->execute([
            'pred' => $result['prediction'],
            'pn' => $result['prob_normal'],
            'pi' => $result['prob_ischemic'],
            'ph' => $result['prob_hemorrhagic'],
            'heatmap' => $heatmapPath,
            'ood' => likely_not_a_scan_value($result),
            'endpoint' => $result['_endpoint'],
            'mv' => isset($result['model_version']) ? substr((string)$result['model_version'], 0, 100) : null,
            'pathway' => isset($result['pathway']) ? substr((string)$result['pathway'], 0, 40) : null,
            'evaluated' => array_key_exists('evaluated_pathway', $result) ? ($result['evaluated_pathway'] ? 1 : 0) : null,
            'warnings' => $warnings !== null ? json_encode(array_values(array_map('strval', $warnings))) : null,
            'note' => !empty($result['decision_note']) ? substr((string)$result['decision_note'], 0, 300) : null,
            'sid' => $scanId,
        ]);
        db()->prepare('UPDATE scans SET status = :status WHERE id = :sid')
            ->execute(['status' => 'completed', 'sid' => $scanId]);
        log_action('prediction_completed', 'scan', (string)$scanId,
            ['prediction' => $result['prediction'], 'pathway' => $result['pathway'] ?? null] + $auditExtra);
        return $result;
    } catch (InferenceError $e) {
        db()->prepare('UPDATE predictions SET attempt_count = attempt_count + 1, last_error = :err WHERE scan_id = :sid')
            ->execute(['err' => $e->getMessage(), 'sid' => $scanId]);
        db()->prepare('UPDATE scans SET status = :status WHERE id = :sid')
            ->execute(['status' => 'failed', 'sid' => $scanId]);
        log_action('prediction_failed', 'scan', (string)$scanId, ['error' => $e->getMessage()] + $auditExtra);
        throw $e;
    }
}

/** Decodes predictions.warnings (JSON array) for display; [] when absent. */
function prediction_warnings(?string $json): array
{
    $w = json_decode((string)$json, true);
    return is_array($w) ? array_values(array_filter($w, 'is_string')) : [];
}

/** Human-readable name of an input pathway, or null when unknown (older predictions). */
function pathway_label(?string $pathway): ?string
{
    return [
        'ct_adjacent_slices' => 'CT with adjacent slices (evaluated pathway)',
        'mri_dwi_adc' => 'MRI DWI + ADC (evaluated pathway)',
        'single_image' => 'Single image (not the evaluated pathway)',
    ][$pathway ?? ''] ?? null;
}
