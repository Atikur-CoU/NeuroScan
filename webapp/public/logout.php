<?php
require_once __DIR__ . '/includes/auth.php';
require_once __DIR__ . '/includes/csrf.php';
require_once __DIR__ . '/includes/audit.php';

bootstrap_session();

// Logging out changes state, so it needs a POST with the CSRF token: a link or image on another site
// cannot log a user out. A plain GET just goes back to the app.
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: ' . (is_logged_in() ? 'index.php' : 'login.php'));
    exit;
}
csrf_check();

if (is_logged_in()) {
    log_action('logout');
    logout_user();
}

header('Location: login.php');
exit;
