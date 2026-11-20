<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();
$user = require_role(['admin']);

$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'assign') {
    csrf_check();
    $patientId = (int)($_POST['patient_id'] ?? 0);
    $providerId = (int)($_POST['provider_id'] ?? 0);
    $relationshipType = in_array($_POST['relationship_type'] ?? '', ['primary_doctor', 'radiologist', 'consulting_doctor'], true)
        ? $_POST['relationship_type'] : 'primary_doctor';

    // Both must be active accounts of the right kind; anything else would fail the foreign key or link the wrong role.
    $roleOf = function (int $id): ?string {
        $stmt = db()->prepare('SELECT role FROM users WHERE id = :id AND is_active = 1');
        $stmt->execute(['id' => $id]);
        $role = $stmt->fetchColumn();
        return $role === false ? null : $role;
    };

    if (!$patientId || !$providerId) {
        $error = 'Please choose both a patient and a provider.';
    } elseif ($roleOf($patientId) !== 'patient' || !in_array($roleOf($providerId), ['doctor', 'radiologist'], true)) {
        $error = 'Please choose an active patient and an active doctor or radiologist.';
    } else {
        $stmt = db()->prepare(
            'SELECT id, is_active FROM doctor_patient WHERE patient_id = :pid AND provider_id = :prid AND relationship_type = :rt'
        );
        $stmt->execute(['pid' => $patientId, 'prid' => $providerId, 'rt' => $relationshipType]);
        $existing = $stmt->fetch();

        if ($existing) {
            db()->prepare('UPDATE doctor_patient SET is_active = 1 WHERE id = :id')->execute(['id' => $existing['id']]);
        } else {
            $stmt = db()->prepare(
                'INSERT INTO doctor_patient (patient_id, provider_id, relationship_type, assigned_by) VALUES (:pid, :prid, :rt, :by)'
            );
            $stmt->execute(['pid' => $patientId, 'prid' => $providerId, 'rt' => $relationshipType, 'by' => $user['id']]);
        }
        log_action('patient_provider_assigned', 'doctor_patient', $patientId . ':' . $providerId, ['relationship_type' => $relationshipType]);
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'unassign') {
    csrf_check();
    $linkId = (int)($_POST['link_id'] ?? 0);
    if ($linkId) {
        $stmt = db()->prepare('SELECT patient_id, provider_id FROM doctor_patient WHERE id = :id');
        $stmt->execute(['id' => $linkId]);
        $link = $stmt->fetch();
        db()->prepare('UPDATE doctor_patient SET is_active = 0 WHERE id = :id')->execute(['id' => $linkId]);
        if ($link) {
            log_action('patient_provider_unassigned', 'doctor_patient', $link['patient_id'] . ':' . $link['provider_id']);
        }
    }
    header('Location: admin_assign.php');
    exit;
}

$patients = db()->query("SELECT id, full_name, email FROM users WHERE role = 'patient' AND is_active = 1 ORDER BY full_name")->fetchAll();
$providers = db()->query("SELECT id, role, full_name, email FROM users WHERE role IN ('doctor','radiologist') AND is_active = 1 ORDER BY full_name")->fetchAll();

$links = db()->query(
    "SELECT dp.id, dp.relationship_type, dp.assigned_at,
            p.full_name AS patient_name, pr.full_name AS provider_name, pr.role AS provider_role
     FROM doctor_patient dp
     JOIN users p ON p.id = dp.patient_id
     JOIN users pr ON pr.id = dp.provider_id
     WHERE dp.is_active = 1
     ORDER BY dp.assigned_at DESC"
)->fetchAll();

$pageTitle = 'Assignments';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Patient &ndash; provider assignments</h1>
  <p class="lede">Link a patient to the doctor(s)/radiologist(s) who should see and review their scans.</p>
</header>

<?php if ($error): ?><div class="flash err"><?= htmlspecialchars($error, ENT_QUOTES) ?></div><?php endif; ?>

<div class="card">
  <h2>New assignment</h2>
  <?php if (!$patients || !$providers): ?>
    <p class="lede">You need at least one patient and one active doctor/radiologist account to create an assignment.</p>
  <?php else: ?>
    <form method="post" action="admin_assign.php">
      <?= csrf_field() ?>
      <input type="hidden" name="action" value="assign">
      <label for="patient_id">Patient</label>
      <select id="patient_id" name="patient_id" required>
        <?php foreach ($patients as $p): ?>
          <option value="<?= (int)$p['id'] ?>"><?= htmlspecialchars($p['full_name'], ENT_QUOTES) ?> (<?= htmlspecialchars($p['email'], ENT_QUOTES) ?>)</option>
        <?php endforeach; ?>
      </select>
      <label for="provider_id">Provider</label>
      <select id="provider_id" name="provider_id" required>
        <?php foreach ($providers as $p): ?>
          <option value="<?= (int)$p['id'] ?>"><?= htmlspecialchars($p['full_name'], ENT_QUOTES) ?> (<?= htmlspecialchars($p['role'], ENT_QUOTES) ?>)</option>
        <?php endforeach; ?>
      </select>
      <label for="relationship_type">Relationship</label>
      <select id="relationship_type" name="relationship_type">
        <option value="primary_doctor">Primary doctor</option>
        <option value="radiologist">Radiologist</option>
        <option value="consulting_doctor">Consulting doctor</option>
      </select>
      <div style="margin-top:20px;"><button type="submit">Assign</button></div>
    </form>
  <?php endif; ?>
</div>

<div class="card">
  <h2>Active assignments</h2>
  <?php if (!$links): ?>
    <p class="lede">No assignments yet.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Patient</th><th>Provider</th><th>Relationship</th><th>Since</th><th></th></tr></thead>
      <tbody>
        <?php foreach ($links as $l): ?>
          <tr>
            <td><?= htmlspecialchars($l['patient_name'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($l['provider_name'], ENT_QUOTES) ?> (<?= htmlspecialchars($l['provider_role'], ENT_QUOTES) ?>)</td>
            <td><?= htmlspecialchars($l['relationship_type'], ENT_QUOTES) ?></td>
            <td><?= htmlspecialchars($l['assigned_at'], ENT_QUOTES) ?></td>
            <td>
              <form method="post" action="admin_assign.php" style="display:inline;">
                <?= csrf_field() ?>
                <input type="hidden" name="action" value="unassign">
                <input type="hidden" name="link_id" value="<?= (int)$l['id'] ?>">
                <button type="submit" class="btn secondary" style="padding:4px 10px; font-size:12px;">Unassign</button>
              </form>
            </td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
