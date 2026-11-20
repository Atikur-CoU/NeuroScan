// High-risk audio cue (moved out of the pages so the CSP can forbid inline script).
(function () {
  // High-risk audio cue -- best-effort only. Browsers may block
  // AudioContext until a user gesture on the page; the try/catch means
  // the visual red badge above is always shown regardless of whether
  // this plays.
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    setTimeout(function () { osc.stop(); ctx.close(); }, 400);
  } catch (e) {}
})();
