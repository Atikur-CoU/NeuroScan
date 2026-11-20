<?php
require_once __DIR__ . '/includes/auth.php';

bootstrap_session();
$user = require_role(['admin']);

const KNOWN_ACTIONS = [
    'login_success', 'login_failed', 'logout', 'account_locked', 'password_changed',
    'user_created', 'user_updated', 'user_deactivated',
    'patient_provider_assigned', 'patient_provider_unassigned',
    'scan_uploaded', 'scan_viewed',
    'prediction_requested', 'prediction_completed', 'prediction_failed',
    'review_created', 'review_finalized', 'review_amended',
    'settings_updated',
];

$actorId = (int)($_GET['actor_id'] ?? 0);
$action = (string)($_GET['action'] ?? '');
$dateFrom = (string)($_GET['date_from'] ?? '');
$dateTo = (string)($_GET['date_to'] ?? '');
$page = max(1, (int)($_GET['page'] ?? 1));
$perPage = 50;

$where = [];
$params = [];
if ($actorId) {
    $where[] = 'al.actor_user_id = :actor_id';
    $params['actor_id'] = $actorId;
}
if ($action !== '' && in_array($action, KNOWN_ACTIONS, true)) {
    $where[] = 'al.action = :action';
    $params['action'] = $action;
}
if ($dateFrom !== '') {
    $where[] = 'al.created_at >= :date_from';
    $params['date_from'] = $dateFrom . ' 00:00:00';
}
if ($dateTo !== '') {
    $where[] = 'al.created_at <= :date_to';
    $params['date_to'] = $dateTo . ' 23:59:59';
}
$whereSql = $where ? ('WHERE ' . implode(' AND ', $where)) : '';

$countStmt = db()->prepare("SELECT COUNT(*) AS c FROM audit_log al $whereSql");
$countStmt->execute($params);
$total = (int)$countStmt->fetch()['c'];
$totalPages = max(1, (int)ceil($total / $perPage));
$page = min($page, $totalPages);
$offset = ($page - 1) * $perPage;

$sql = "SELECT al.*, u.full_name AS actor_name
        FROM audit_log al LEFT JOIN users u ON u.id = al.actor_user_id
        $whereSql
        ORDER BY al.id DESC
        LIMIT :limit OFFSET :offset";
$stmt = db()->prepare($sql);
foreach ($params as $key => $value) {
    $stmt->bindValue($key, $value);
}
$stmt->bindValue('limit', $perPage, PDO::PARAM_INT);
$stmt->bindValue('offset', $offset, PDO::PARAM_INT);
$stmt->execute();
$rows = $stmt->fetchAll();

$users = db()->query('SELECT id, full_name, role FROM users ORDER BY full_name')->fetchAll();

function qs(array $overrides = []): string
{
    $params = array_merge($_GET, $overrides);
    return htmlspecialchars('?' . http_build_query($params), ENT_QUOTES);
}

$pageTitle = 'Audit log';
require __DIR__ . '/includes/layout_header.php';
?>
<header class="page-header">
  <span class="eyebrow">NeuroScan &middot; Admin</span>
  <h1>Audit log</h1>
  <p class="lede"><?= number_format($total) ?> events<?= $where ? ' matching the filters below' : '' ?>.</p>
</header>

<div class="card">
  <form method="get" action="admin_audit_log.php">
    <label for="actor_id">Actor</label>
    <select id="actor_id" name="actor_id">
      <option value="0">All users</option>
      <?php foreach ($users as $u): ?>
        <option value="<?= (int)$u['id'] ?>" <?= $actorId === (int)$u['id'] ? 'selected' : '' ?>>
          <?= htmlspecialchars($u['full_name'], ENT_QUOTES) ?> (<?= htmlspecialchars($u['role'], ENT_QUOTES) ?>)
        </option>
      <?php endforeach; ?>
    </select>
    <label for="action">Action</label>
    <select id="action" name="action">
      <option value="">All actions</option>
      <?php foreach (KNOWN_ACTIONS as $a): ?>
        <option value="<?= htmlspecialchars($a, ENT_QUOTES) ?>" <?= $action === $a ? 'selected' : '' ?>><?= htmlspecialchars($a, ENT_QUOTES) ?></option>
      <?php endforeach; ?>
    </select>
    <label for="date_from">From</label>
    <input type="date" id="date_from" name="date_from" value="<?= htmlspecialchars($dateFrom, ENT_QUOTES) ?>">
    <label for="date_to">To</label>
    <input type="date" id="date_to" name="date_to" value="<?= htmlspecialchars($dateTo, ENT_QUOTES) ?>">
    <div style="margin-top:20px; display:flex; gap:10px;">
      <button type="submit">Filter</button>
      <a class="btn secondary" href="admin_audit_log.php">Clear</a>
    </div>
  </form>
</div>

<div class="card" style="overflow-x:auto;">
  <?php if (!$rows): ?>
    <p class="lede">No matching events.</p>
  <?php else: ?>
    <table>
      <thead><tr><th>Time (UTC)</th><th>Actor</th><th>Action</th><th>Target</th><th>Details</th></tr></thead>
      <tbody>
        <?php foreach ($rows as $r): ?>
          <tr>
            <td><?= htmlspecialchars($r['created_at'], ENT_QUOTES) ?></td>
            <td><?= $r['actor_name'] ? htmlspecialchars($r['actor_name'], ENT_QUOTES) . ' (' . htmlspecialchars((string)$r['actor_role'], ENT_QUOTES) . ')' : '&mdash;' ?></td>
            <td><code><?= htmlspecialchars($r['action'], ENT_QUOTES) ?></code></td>
            <td><?= $r['target_type'] ? htmlspecialchars($r['target_type'] . ':' . $r['target_id'], ENT_QUOTES) : '&mdash;' ?></td>
            <td style="max-width:280px; overflow-wrap:break-word; font-size:12px; color:var(--text-faint);"><?= $r['details'] ? htmlspecialchars($r['details'], ENT_QUOTES) : '' ?></td>
          </tr>
        <?php endforeach; ?>
      </tbody>
    </table>
  <?php endif; ?>
</div>

<?php if ($totalPages > 1): ?>
  <div style="display:flex; gap:10px; align-items:center; margin-top:10px;">
    <?php if ($page > 1): ?><a class="btn secondary" href="<?= qs(['page' => $page - 1]) ?>">&larr; Newer</a><?php endif; ?>
    <span class="lede">Page <?= $page ?> of <?= $totalPages ?></span>
    <?php if ($page < $totalPages): ?><a class="btn secondary" href="<?= qs(['page' => $page + 1]) ?>">Older &rarr;</a><?php endif; ?>
  </div>
<?php endif; ?>

<?php require __DIR__ . '/includes/layout_footer.php'; ?>
