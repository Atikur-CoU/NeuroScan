<?php
require_once __DIR__ . '/config.php';

class UploadValidationError extends Exception {}

// Generous for a real CT/MRI slice (typically well under 1024px per side)
// while still bounding worst-case decoded memory use.
const MAX_IMAGE_DIMENSION = 4096;

/**
 * Validates an uploaded image, re-encodes it (strips EXIF/embedded payloads,
 * normalizes format), and writes it to UPLOADS_DIR under a random filename.
 * Returns [stored_filename, storage_path (relative to UPLOADS_DIR), file_size_bytes, mime_type, sha256_hash].
 */
function validate_and_store_upload(array $file, int $maxBytes): array
{
    if (!isset($file['error']) || is_array($file['error'])) {
        throw new UploadValidationError('Invalid upload.');
    }
    if ($file['error'] !== UPLOAD_ERR_OK) {
        throw new UploadValidationError('Upload failed (error code ' . $file['error'] . ').');
    }
    if ($file['size'] > $maxBytes) {
        throw new UploadValidationError('File is too large (limit ' . round($maxBytes / 1024 / 1024) . ' MB).');
    }

    $finfo = finfo_open(FILEINFO_MIME_TYPE);
    $detectedMime = finfo_file($finfo, $file['tmp_name']);
    finfo_close($finfo);

    $allowed = ['image/png' => 'png', 'image/jpeg' => 'jpg'];
    if (!isset($allowed[$detectedMime])) {
        throw new UploadValidationError('Only PNG or JPG images are accepted.');
    }

    $imageInfo = @getimagesize($file['tmp_name']);
    if ($imageInfo === false) {
        throw new UploadValidationError('Could not read this file as an image.');
    }
    // Reject oversized dimensions before decoding -- a small, highly
    // compressed file (e.g. a huge solid-color PNG) can still decode into
    // a multi-gigabyte pixel buffer ("decompression bomb"). getimagesize()
    // reads only the header, so this check is cheap even for a hostile file.
    [$width, $height] = $imageInfo;
    if ($width > MAX_IMAGE_DIMENSION || $height > MAX_IMAGE_DIMENSION) {
        throw new UploadValidationError(
            "Image dimensions too large ({$width}x{$height}, limit " . MAX_IMAGE_DIMENSION . "px per side)."
        );
    }

    switch ($detectedMime) {
        case 'image/png':
            $img = @imagecreatefrompng($file['tmp_name']);
            break;
        case 'image/jpeg':
            $img = @imagecreatefromjpeg($file['tmp_name']);
            break;
        default:
            $img = false;
    }
    if ($img === false) {
        throw new UploadValidationError('Could not decode this image.');
    }

    if (!is_dir(UPLOADS_DIR)) {
        mkdir(UPLOADS_DIR, 0750, true);
    }

    $storedFilename = bin2hex(random_bytes(16)) . '.png';
    $fullPath = UPLOADS_DIR . '/' . $storedFilename;

    // Re-encode as PNG regardless of source format -- strips EXIF/embedded
    // metadata and normalizes what the inference endpoint receives.
    imagesavealpha($img, true);
    $ok = imagepng($img, $fullPath, 6);
    imagedestroy($img);
    if (!$ok) {
        throw new UploadValidationError('Failed to store the uploaded image.');
    }

    return [
        'stored_filename' => $storedFilename,
        'storage_path' => $storedFilename,
        'file_size_bytes' => filesize($fullPath),
        'mime_type' => 'image/png',
        'sha256_hash' => hash_file('sha256', $fullPath),
    ];
}
