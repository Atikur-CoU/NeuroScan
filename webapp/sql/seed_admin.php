<?php
/**
 * One-time bootstrap for the first admin account. Run this from the CLI
 * (`php sql/seed_admin.php`) once, right after importing schema.sql --
 * there is no web UI path to create the first admin, by design (nothing
 * exists yet to gate that privilege). Delete or move this file, or at
 * least don't leave it reachable over HTTP, once you've run it.
 */
require_once __DIR__ . '/../includes/db.php';

function prompt(string $label, bool $hidden = false): string
{
    echo $label;
    if ($hidden && stripos(PHP_OS, 'WIN') === false) {
        system('stty -echo');
        $value = trim((string)fgets(STDIN));
        system('stty echo');
        echo "\n";
    } else {
        $value = trim((string)fgets(STDIN));
    }
    return $value;
}

echo "NeuroScan -- create the first admin account\n";
$name = prompt('Full name: ');
$email = prompt('Email: ');
$password = prompt('Password (min 8 chars): ', true);

if (strlen($password) < 8) {
    fwrite(STDERR, "Password must be at least 8 characters.\n");
    exit(1);
}

$stmt = db()->prepare('SELECT id FROM users WHERE email = :email');
$stmt->execute(['email' => $email]);
if ($stmt->fetch()) {
    fwrite(STDERR, "A user with that email already exists.\n");
    exit(1);
}

$stmt = db()->prepare(
    'INSERT INTO users (role, full_name, email, password_hash) VALUES (:role, :name, :email, :hash)'
);
$stmt->execute([
    'role' => 'admin',
    'name' => $name,
    'email' => $email,
    'hash' => password_hash($password, PASSWORD_DEFAULT),
]);

echo "Admin account created (id " . db()->lastInsertId() . "). You can log in at /login.php.\n";
