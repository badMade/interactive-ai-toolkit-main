// Popup script manages user interaction, audio capture, playback, and messaging with the background worker.
const browserApi = typeof browser !== 'undefined' ? browser : chrome;
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const elements = {
  apiKey: document.getElementById('api-key'),
  passphrase: document.getElementById('passphrase'),
  saveKey: document.getElementById('save-key'),
  removeKey: document.getElementById('remove-key'),
  keyStatus: document.getElementById('key-status'),
  voice: document.getElementById('voice-select'),
  ttsFormat: document.getElementById('tts-format'),
  inputLanguage: document.getElementById('input-language'),
  outputLanguage: document.getElementById('output-language'),
  summariesToggle: document.getElementById('summaries-toggle'),
  transcriptionToggle: document.getElementById('transcription-toggle'),
  monthlyLimit: document.getElementById('monthly-limit'),
  clearCache: document.getElementById('clear-cache'),
  startReading: document.getElementById('start-reading'),
  stopReading: document.getElementById('stop-reading'),
  pushToTalk: document.getElementById('push-to-talk'),
  transcript: document.getElementById('transcript-text'),
  questionInput: document.getElementById('question-input'),
  askQuestion: document.getElementById('ask-question'),
  usageTranscription: document.getElementById('usage-transcription'),
  usageTts: document.getElementById('usage-tts'),
  usageChat: document.getElementById('usage-chat'),
  usageCost: document.getElementById('usage-cost'),
  player: document.getElementById('player')
};

const portId = crypto.randomUUID();
const ttsPort = browserApi.runtime.connect({ name: 'tts-stream' });
ttsPort.postMessage({ type: 'REGISTER_PORT', portId });

let mediaSource = null;
let sourceBuffer = null;
let audioQueue = [];
let currentFormat = 'mp3';
let mediaRecorder = null;
let recorderStream = null;
let recordingStartedAt = null;
let shouldFinalizeStream = false;
let mediaSourceUrl = null;
let fallbackChunks = null;

const settingsState = {
  hasApiKey: false
};

init().catch((error) => {
  console.error(error);
  renderStatus('Failed to initialize popup. See console for details.', true);
});

ttsPort.onMessage.addListener((message) => {
  switch (message?.type) {
    case 'tts-start':
      prepareMediaSource(message.format ?? 'mp3');
      break;
    case 'tts-chunk':
      if (message.chunk) {
        enqueueChunk(message.chunk);
      }
      break;
    case 'tts-end':
      finalizeStream();
      break;
    default:
      break;
  }
});

elements.saveKey.addEventListener('click', async () => {
  try {
    await saveApiKey();
    renderStatus('API key encrypted and saved.', false);
    settingsState.hasApiKey = true;
  } catch (error) {
    renderStatus(error.message, true);
  }
});

elements.removeKey.addEventListener('click', async () => {
  try {
    await browserApi.runtime.sendMessage({ type: 'REMOVE_API_KEY' });
    renderStatus('API key removed.', false);
    settingsState.hasApiKey = false;
    elements.apiKey.value = '';
  } catch (error) {
    renderStatus(error.message, true);
  }
});

elements.startReading.addEventListener('click', async () => {
  try {
    await persistSettings();
    await requirePassphrase();
    elements.startReading.disabled = true;
    elements.stopReading.disabled = false;
    renderStatus('Preparing narration…', false);
    const response = await browserApi.runtime.sendMessage({
      type: 'REQUEST_PAGE_READ',
      portId,
      voice: elements.voice.value,
      ttsFormat: elements.ttsFormat.value,
      targetLanguage: elements.outputLanguage.value || 'en',
      summarizationEnabled: elements.summariesToggle.checked
    });
    if (!response?.ok) {
      throw new Error(response?.error ?? 'Failed to start reading');
    }
  } catch (error) {
    elements.startReading.disabled = false;
    elements.stopReading.disabled = true;
    renderStatus(error.message, true);
  }
});

elements.stopReading.addEventListener('click', () => {
  browserApi.runtime
    .sendMessage({ type: 'STOP_PAGE_READ', portId })
    .catch((error) => console.debug('Failed to cancel narration', error));
  stopPlayback();
  elements.startReading.disabled = false;
  elements.stopReading.disabled = true;
  renderStatus('Playback stopped.', false);
});

elements.askQuestion.addEventListener('click', async () => {
  const question = elements.questionInput.value.trim();
  if (!question) {
    renderStatus('Enter a question first.', true);
    return;
  }
  try {
    await persistSettings();
    await requirePassphrase();
    renderStatus('Asking assistant…', false);
    const response = await browserApi.runtime.sendMessage({
      type: 'ASK_QUESTION',
      question,
      context: '',
      modelPreference: null
    });
    if (!response?.ok) {
      throw new Error(response?.error ?? 'Failed to ask question');
    }
    renderStatus('Answer ready.', false);
    elements.questionInput.value = '';
    elements.transcript.textContent = response.result;
    await refreshUsage();
  } catch (error) {
    renderStatus(error.message, true);
  }
});

elements.clearCache.addEventListener('click', async () => {
  try {
    await browserApi.runtime.sendMessage({ type: 'CLEAR_CACHE' });
    renderStatus('Caches cleared.', false);
  } catch (error) {
    renderStatus(error.message, true);
  }
});

[
  elements.summariesToggle,
  elements.transcriptionToggle,
  elements.voice,
  elements.ttsFormat,
  elements.inputLanguage,
  elements.outputLanguage,
  elements.monthlyLimit
].forEach((control) => {
  control?.addEventListener('change', () => {
    persistSettings().catch((error) => renderStatus(error.message, true));
  });
});

elements.pushToTalk.addEventListener('mousedown', startRecording);
elements.pushToTalk.addEventListener('touchstart', startRecording);
elements.pushToTalk.addEventListener('mouseup', stopRecording);
elements.pushToTalk.addEventListener('mouseleave', stopRecording);
elements.pushToTalk.addEventListener('touchend', stopRecording);

async function init() {
  // Restore persisted preferences and usage meters when the popup opens.
  await loadSettings();
  await refreshUsage();
}

async function loadSettings() {
  const response = await browserApi.runtime.sendMessage({ type: 'LOAD_SETTINGS' });
  if (!response?.ok) {
    throw new Error(response?.error ?? 'Unable to load settings');
  }
  const { settings, hasApiKey } = response.result;
  settingsState.hasApiKey = hasApiKey;
  elements.voice.value = settings.voice ?? 'alloy';
  elements.ttsFormat.value = settings.ttsFormat ?? 'mp3';
  elements.inputLanguage.value = settings.inputLanguage ?? 'en';
  elements.outputLanguage.value = settings.outputLanguage ?? 'en';
  elements.summariesToggle.checked = Boolean(settings.summarizationEnabled);
  elements.transcriptionToggle.checked = Boolean(settings.transcriptionForLongPages);
  elements.monthlyLimit.value = settings.monthlyLimit ?? 10;
  if (!hasApiKey) {
    renderStatus('Add and encrypt your API key to get started.', false);
  }
}

async function persistSettings() {
  const settings = {
    voice: elements.voice.value,
    ttsFormat: elements.ttsFormat.value,
    inputLanguage: elements.inputLanguage.value || 'en',
    outputLanguage: elements.outputLanguage.value || 'en',
    summarizationEnabled: elements.summariesToggle.checked,
    transcriptionForLongPages: elements.transcriptionToggle.checked,
    monthlyLimit: Number(elements.monthlyLimit.value) || 0
  };
  const response = await browserApi.runtime.sendMessage({ type: 'SAVE_SETTINGS', settings });
  if (!response?.ok) {
    throw new Error(response?.error ?? 'Failed to save settings');
  }
  return response.result;
}

function renderStatus(message, isError) {
  elements.keyStatus.textContent = message;
  elements.keyStatus.classList.toggle('error', Boolean(isError));
}

async function saveApiKey() {
  const apiKey = elements.apiKey.value.trim();
  const passphrase = elements.passphrase.value.trim();
  if (!apiKey) {
    throw new Error('Enter your OpenAI API key.');
  }
  if (!passphrase) {
    throw new Error('Provide a passphrase for encryption.');
  }
  const response = await browserApi.runtime.sendMessage({ type: 'SAVE_API_KEY', apiKey, passphrase });
  if (!response?.ok) {
    throw new Error(response?.error ?? 'Failed to save API key');
  }
  elements.apiKey.value = '';
  elements.passphrase.value = '';
}

async function requirePassphrase() {
  if (!settingsState.hasApiKey) {
    return;
  }
  const passphrase = elements.passphrase.value.trim();
  if (!passphrase) {
    throw new Error('Enter your passphrase to unlock the key.');
  }
  const response = await browserApi.runtime.sendMessage({ type: 'SET_PASSPHRASE', passphrase });
  if (!response?.ok) {
    throw new Error(response?.error ?? 'Failed to unlock API key');
  }
  if (!response.result?.success) {
    throw new Error(response.result?.error ?? 'Passphrase rejected. Re-enter to unlock.');
  }
}

function prepareMediaSource(format) {
  cleanupMediaSource();
  currentFormat = format;
  if (!('MediaSource' in window)) {
    renderStatus('Streaming playback is not supported in this browser. Buffering audio…', true);
    fallbackChunks = [];
    return;
  }
  mediaSource = new MediaSource();
  audioQueue = [];
  shouldFinalizeStream = false;
  if (mediaSourceUrl) {
    URL.revokeObjectURL(mediaSourceUrl);
  }
  mediaSourceUrl = URL.createObjectURL(mediaSource);
  elements.player.src = mediaSourceUrl;
  mediaSource.addEventListener('sourceopen', () => {
    try {
      const mimeType = format === 'wav' ? 'audio/wav; codecs="1"' : 'audio/mpeg';
      sourceBuffer = mediaSource.addSourceBuffer(mimeType);
      sourceBuffer.addEventListener('updateend', handleSourceBufferUpdate);
      flushQueue();
    } catch (error) {
      renderStatus('Audio format not supported: ' + error.message, true);
    }
  });
}

function enqueueChunk(chunkBuffer) {
  const chunk = new Uint8Array(chunkBuffer);
  if (fallbackChunks) {
    fallbackChunks.push(chunk);
    return;
  }
  audioQueue.push(chunk);
  flushQueue();
}

function flushQueue() {
  if (!sourceBuffer || sourceBuffer.updating || audioQueue.length === 0) {
    return;
  }
  const chunk = audioQueue.shift();
  try {
    sourceBuffer.appendBuffer(chunk);
    if (elements.player.paused) {
      elements.player.play().catch(() => {});
    }
  } catch (error) {
    console.error('Failed to append audio chunk', error);
  }
}

function handleSourceBufferUpdate() {
  flushQueue();
  if (shouldFinalizeStream && (!sourceBuffer || (!sourceBuffer.updating && audioQueue.length === 0))) {
    finalizeStream();
  }
}

function finalizeStream() {
  if (sourceBuffer && sourceBuffer.updating) {
    shouldFinalizeStream = true;
    return;
  }
  shouldFinalizeStream = false;
  if (fallbackChunks && fallbackChunks.length) {
    const blob = new Blob(fallbackChunks, { type: currentFormat === 'wav' ? 'audio/wav' : 'audio/mpeg' });
    fallbackChunks = null;
    if (mediaSourceUrl) {
      URL.revokeObjectURL(mediaSourceUrl);
    }
    mediaSourceUrl = URL.createObjectURL(blob);
    elements.player.src = mediaSourceUrl;
    elements.player.play().catch(() => {});
  }
  if (mediaSource && mediaSource.readyState === 'open') {
    try {
      mediaSource.endOfStream();
    } catch (error) {
      console.debug('Unable to end stream', error);
    }
  }
  sourceBuffer?.removeEventListener('updateend', handleSourceBufferUpdate);
  sourceBuffer = null;
  audioQueue = [];
  refreshUsage().catch((error) => console.error(error));
  elements.startReading.disabled = false;
  elements.stopReading.disabled = true;
  renderStatus('Narration finished.', false);
}

function cleanupMediaSource() {
  shouldFinalizeStream = false;
  if (sourceBuffer) {
    try {
      sourceBuffer.removeEventListener('updateend', handleSourceBufferUpdate);
    } catch (error) {
      console.debug(error);
    }
  }
  if (mediaSource) {
    try {
      if (mediaSource.readyState === 'open') {
        mediaSource.endOfStream();
      }
    } catch (error) {
      console.debug(error);
    }
  }
  if (mediaSourceUrl) {
    URL.revokeObjectURL(mediaSourceUrl);
    mediaSourceUrl = null;
  }
  mediaSource = null;
  sourceBuffer = null;
  audioQueue = [];
  fallbackChunks = null;
}

function stopPlayback() {
  cleanupMediaSource();
  elements.player.pause();
  elements.player.currentTime = 0;
}

async function refreshUsage() {
  const response = await browserApi.runtime.sendMessage({ type: 'LOAD_USAGE' });
  if (!response?.ok) {
    renderStatus(response?.error ?? 'Failed to load usage', true);
    return;
  }
  const usage = response.result;
  const minutes = (usage.transcriptionSeconds ?? 0) / 60;
  elements.usageTranscription.textContent = minutes.toFixed(2);
  elements.usageTts.textContent = (usage.ttsCharacters ?? 0).toLocaleString();
  elements.usageChat.textContent = (usage.chatTokens ?? 0).toLocaleString();
  elements.usageCost.textContent = Number(usage.estimatedCost ?? 0).toFixed(2);
  const limit = Number(elements.monthlyLimit.value) || 0;
  if (limit && usage.estimatedCost > limit) {
    renderStatus('Monthly budget exceeded. Consider lowering usage.', true);
  }
}

async function startRecording(event) {
  event.preventDefault();
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    return;
  }
  if (typeof MediaRecorder === 'undefined') {
    renderStatus('MediaRecorder is not supported in this browser.', true);
    return;
  }
  try {
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    renderStatus('Microphone access denied.', true);
    elements.pushToTalk.setAttribute('aria-pressed', 'false');
    return;
  }
  const mimeType = getSupportedMimeType();
  try {
    mediaRecorder = new MediaRecorder(recorderStream, { mimeType });
  } catch (error) {
    renderStatus('Recording format not supported.', true);
    recorderStream.getTracks().forEach((track) => track.stop());
    recorderStream = null;
    elements.pushToTalk.setAttribute('aria-pressed', 'false');
    return;
  }
  const recorded = [];
  recordingStartedAt = Date.now();
  mediaRecorder.addEventListener('dataavailable', (eventData) => {
    if (eventData.data.size > 0) {
      recorded.push(eventData.data);
    }
  });
  mediaRecorder.addEventListener('stop', async () => {
    elements.pushToTalk.classList.remove('recording');
    elements.pushToTalk.setAttribute('aria-pressed', 'false');
    recorderStream?.getTracks().forEach((track) => track.stop());
    recorderStream = null;
    try {
      await submitTranscription(recorded, mimeType);
    } catch (error) {
      renderStatus(error.message, true);
    }
    mediaRecorder = null;
  });
  elements.pushToTalk.classList.add('recording');
  elements.pushToTalk.setAttribute('aria-pressed', 'true');
  mediaRecorder.start(500);
}

function stopRecording(event) {
  event?.preventDefault?.();
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    return;
  }
  mediaRecorder.stop();
}

async function submitTranscription(blobs, mimeType) {
  // Compresses the recording payload and forwards it to the background worker for Whisper transcription.
  if (!blobs.length) {
    return;
  }
  await requirePassphrase();
  const totalSize = blobs.reduce((sum, blob) => sum + blob.size, 0);
  const buffers = [];
  if (totalSize > MAX_UPLOAD_BYTES) {
    const chunks = splitBlob(blobs, MAX_UPLOAD_BYTES);
    for (const chunk of chunks) {
      buffers.push(await blobToArrayBuffer(chunk));
    }
  } else {
    const combined = new Blob(blobs, { type: mimeType });
    buffers.push(await blobToArrayBuffer(combined));
  }
  const durationSeconds = recordingStartedAt ? (Date.now() - recordingStartedAt) / 1000 : 0;
  const response = await browserApi.runtime.sendMessage({
    type: 'TRANSCRIBE_AUDIO',
    audioChunks: buffers,
    mimeType,
    language: elements.inputLanguage.value || 'en',
    durationSeconds
  });
  if (!response?.ok) {
    throw new Error(response?.error ?? 'Transcription failed');
  }
  const transcript = response.result;
  elements.transcript.textContent = transcript || 'No speech detected.';
  recordingStartedAt = null;
  await refreshUsage();
  if (transcript) {
    await handleCommand(transcript);
  }
}

function splitBlob(blobs, maxBytes) {
  const combined = new Blob(blobs, { type: blobs[0]?.type });
  const result = [];
  let offset = 0;
  while (offset < combined.size) {
    const chunk = combined.slice(offset, offset + maxBytes);
    result.push(chunk);
    offset += maxBytes;
  }
  return result;
}

async function blobToArrayBuffer(blob) {
  const buffer = await blob.arrayBuffer();
  return buffer;
}

async function handleCommand(transcript) {
  // Lightweight command parser for hands-free navigation and controls.
  const lower = transcript.toLowerCase();
  if (lower.includes('scroll down')) {
    await executeTabAction({ action: 'scroll', direction: 'down' });
    renderStatus('Scrolling down…', false);
    return;
  }
  if (lower.includes('scroll up')) {
    await executeTabAction({ action: 'scroll', direction: 'up' });
    renderStatus('Scrolling up…', false);
    return;
  }
  if (lower.includes('read page')) {
    elements.startReading.click();
    return;
  }
  if (lower.startsWith('summarize')) {
    elements.questionInput.value = transcript;
    elements.askQuestion.click();
    return;
  }
  if (lower.includes('stop')) {
    elements.stopReading.click();
  }
}

async function executeTabAction(command) {
  // Delegates page control commands to the content script for execution in the active tab.
  const tabs = await browserApi.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length) {
    return;
  }
  await browserApi.tabs.sendMessage(tabs[0].id, { type: 'CONTROL_ACTION', command }).catch(() => {});
}

function getSupportedMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/wav'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || 'audio/webm';
}

function cleanupRecorder() {
  recorderStream?.getTracks().forEach((track) => track.stop());
  recorderStream = null;
  mediaRecorder = null;
}

window.addEventListener('beforeunload', () => {
  cleanupMediaSource();
  cleanupRecorder();
});
