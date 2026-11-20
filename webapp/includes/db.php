<?php
require_once __DIR__ . '/config.php';

// Every page loads this file. Errors go to the server log, never into the page: a database fault would otherwise
// show file paths and query details to visitors on a host whose PHP displays errors.
ini_set('display_errors', '0');
ini_set('log_errors', '1');

function db(): PDO
{
    static $pdo = null;
    if ($pdo === null) {
        $dsn = 'mysql:host=' . DB_HOST . ';dbname=' . DB_NAME . ';charset=utf8mb4';
        $pdo = new PDO($dsn, DB_USER, DB_PASS, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES => false,
        ]);
    }
    return $pdo;
}
