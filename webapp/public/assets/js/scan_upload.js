// Upload page: drop zone, preview and the CT/MRI context fields (moved out of the page so the CSP can forbid inline script).
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const previewRow = document.getElementById('preview-row');
const previewImg = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const previewSub = document.getElementById('preview-sub');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showPreview(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) showPreview(fileInput.files[0]);
});
const modalitySel = document.getElementById('modality');
function syncContextFields() {
  const mri = modalitySel.value === 'MRI';
  document.getElementById('ctx-ct').style.display = mri ? 'none' : '';
  document.getElementById('ctx-mri').style.display = mri ? '' : 'none';
  document.getElementById('adc-file').required = mri;
  document.getElementById('main-label').textContent = mri ? 'Image (MRI: the DWI slice)' : 'Image (CT: the slice to assess)';
  // Files chosen for the other modality are cleared so they are never sent.
  (mri ? ['prev-file', 'next-file'] : ['adc-file']).forEach(id => { document.getElementById(id).value = ''; });
}
modalitySel.addEventListener('change', syncContextFields);
syncContextFields();
function showPreview(file) {
  previewImg.src = URL.createObjectURL(file);
  previewName.textContent = file.name;
  previewSub.textContent = `${(file.size / 1024).toFixed(0)} KB`;
  previewRow.classList.add('show');
}
