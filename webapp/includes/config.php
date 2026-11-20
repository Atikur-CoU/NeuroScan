<?php
// Local/dev defaults. On the real host, copy this file outside version control
// and fill in the real DB credentials -- never commit real secrets here.

define('DB_HOST', getenv('NEUROSCAN_DB_HOST') ?: '127.0.0.1');
define('DB_NAME', getenv('NEUROSCAN_DB_NAME') ?: 'neuroscan');
define('DB_USER', getenv('NEUROSCAN_DB_USER') ?: 'root');
define('DB_PASS', getenv('NEUROSCAN_DB_PASS') ?: '');

define('STORAGE_DIR', dirname(__DIR__) . '/storage');
define('UPLOADS_DIR', STORAGE_DIR . '/uploads');
define('LOG_DIR', STORAGE_DIR . '/logs');

// Set to true once the site is confirmed to be served over HTTPS.
define('FORCE_HTTPS_COOKIES', false);
