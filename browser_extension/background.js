// Background service worker: stores encrypted API credentials, coordinates
// network calls, manages caches, and enforces user-defined cost controls.
const browserApi = typeof browser !== 'undefined' ? browser : chrome;

const OPENAI_BASE_URL = 'https://api.openai.com/v1';
const STORAGE_KEYS = {
  SETTINGS: 'sr_settings',
  ENCRYPTED_KEY: 'sr_encrypted_key',
  USAGE: 'sr_usage',
  CACHE: 'sr_cache'
};

const DEFAULT_SETTINGS = {
  preferredModel: 'gpt-3.5-turbo',
  allowGpt4o: false,
  voice: 'alloy',
  ttsFormat: 'mp3',
  inputLanguage: 'en',
  outputLanguage: 'en',
  monthlyLimit: 10,
  summarizationEnabled: true,
  summarizationThreshold: 6000,
  autoScroll: false,
  transcriptionForLongPages: true
};

// Conservative cost estimates used for proactive budget enforcement.
const COST_ESTIMATES = {
  transcriptionPerSecond: 0.006 / 60,
  gpt35Per1kTokens: 0.0005,
  gpt4oPer1kTokens: 0.01,
  ttsPerCharacter: 0.00002
};

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const MAX_CACHE_ENTRIES = 25;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

let sessionState = {
  passphrase: null
};

// Active ports streaming audio back to the popup UI.
const audioPorts = new Map();
const activeNarrations = new Map();

browserApi.runtime.onInstalled.addListener(async () => {
  await ensureDefaults();
});

browserApi.runtime.onStartup?.addListener(async () => {
  await ensureDefaults();
});

browserApi.runtime.onConnect.addListener((port) => {
  if (port.name !== 'tts-stream') {
    return;
  }
  port.onMessage.addListener((message) => {
    if (message?.type === 'REGISTER_PORT' && message?.portId) {
      audioPorts.set(message.portId, port);
      port._portId = message.portId;
    }
  });
  port.onDisconnect.addListener(() => {
    if (port._portId) {
      audioPorts.delete(port._portId);
    }
  });
});

browserApi.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const handler = getHandler(message?.type);
  if (!handler) {
    return false;
  }
  handler(message, sender)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => {
      console.error('Background error', error);
      sendResponse({ ok: false, error: error?.message ?? 'Unknown error' });
    });
  return true;
});

function getHandler(type) {
  const handlers = {
    SAVE_API_KEY: handleSaveApiKey,
    LOAD_SETTINGS: handleLoadSettings,
    SAVE_SETTINGS: handleSaveSettings,
    REQUEST_PAGE_READ: handleRequestPageRead,
    STOP_PAGE_READ: handleStopPageRead,
    TRANSCRIBE_AUDIO: handleTranscribeAudio,
    ASK_QUESTION: handleAskQuestion,
    LOAD_USAGE: handleLoadUsage,
    CLEAR_CACHE: handleClearCache,
    SET_PASSPHRASE: handleSetPassphrase,
    REMOVE_API_KEY: handleRemoveApiKey
  };
  return handlers[type] ?? null;
}

async function ensureDefaults() {
  const storedSettings = await browserApi.storage.local.get([STORAGE_KEYS.SETTINGS, STORAGE_KEYS.USAGE, STORAGE_KEYS.CACHE]);
  if (!storedSettings[STORAGE_KEYS.SETTINGS]) {
    await browserApi.storage.local.set({ [STORAGE_KEYS.SETTINGS]: DEFAULT_SETTINGS });
  }
  if (!storedSettings[STORAGE_KEYS.USAGE]) {
    await browserApi.storage.local.set({
      [STORAGE_KEYS.USAGE]: {
        periodStart: new Date().toISOString(),
        transcriptionSeconds: 0,
        ttsCharacters: 0,
        chatTokens: 0,
        estimatedCost: 0
      }
    });
  }
  if (!storedSettings[STORAGE_KEYS.CACHE]) {
    await browserApi.storage.local.set({
      [STORAGE_KEYS.CACHE]: {
        summaries: {},
        audio: {}
      }
    });
  }
}

async function handleSaveApiKey(message) {
  const { apiKey, passphrase } = message ?? {};
  if (!apiKey) {
    throw new Error('API key is required');
  }
  if (!passphrase) {
    throw new Error('A passphrase is required to encrypt the API key');
  }
  const record = await encryptSecret(apiKey, passphrase);
  await browserApi.storage.local.set({ [STORAGE_KEYS.ENCRYPTED_KEY]: record });
  sessionState.passphrase = passphrase;
  return true;
}

async function handleRemoveApiKey() {
  sessionState.passphrase = null;
  await browserApi.storage.local.remove(STORAGE_KEYS.ENCRYPTED_KEY);
  return true;
}

async function handleSetPassphrase(message) {
  const { passphrase } = message ?? {};
  if (!passphrase) {
    sessionState.passphrase = null;
    return { success: false };
  }
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.ENCRYPTED_KEY);
  const record = stored[STORAGE_KEYS.ENCRYPTED_KEY];
  if (!record) {
    sessionState.passphrase = passphrase;
    return { success: false };
  }
  try {
    await decryptSecret(record, passphrase);
    sessionState.passphrase = passphrase;
    return { success: true };
  } catch (error) {
    console.warn('Failed to unlock API key', error);
    sessionState.passphrase = null;
    return { success: false, error: 'Invalid passphrase' };
  }
}

async function hasStoredApiKey() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.ENCRYPTED_KEY);
  return Boolean(stored[STORAGE_KEYS.ENCRYPTED_KEY]);
}

async function handleLoadSettings() {
  const stored = await browserApi.storage.local.get([STORAGE_KEYS.SETTINGS, STORAGE_KEYS.ENCRYPTED_KEY]);
  const settings = stored[STORAGE_KEYS.SETTINGS] ?? DEFAULT_SETTINGS;
  const hasApiKey = Boolean(stored[STORAGE_KEYS.ENCRYPTED_KEY]);
  return { settings, hasApiKey };
}

async function handleSaveSettings(message) {
  const { settings } = message ?? {};
  if (!settings) {
    throw new Error('Missing settings payload');
  }
  const merged = { ...DEFAULT_SETTINGS, ...settings };
  await browserApi.storage.local.set({ [STORAGE_KEYS.SETTINGS]: merged });
  return merged;
}

async function handleLoadUsage() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.USAGE);
  return stored[STORAGE_KEYS.USAGE];
}

async function handleClearCache() {
  await browserApi.storage.local.set({
    [STORAGE_KEYS.CACHE]: { summaries: {}, audio: {} }
  });
  return true;
}

async function handleTranscribeAudio(message) {
  // Sends audio data to Whisper for speech-to-text while respecting upload limits.
  const { audioChunks, mimeType, language, durationSeconds } = message ?? {};
  if (!Array.isArray(audioChunks) || audioChunks.length === 0) {
    throw new Error('No audio provided');
  }
  const settings = await loadSettings();
  if (!(settings.transcriptionForLongPages ?? true) && (durationSeconds ?? 0) > 60) {
    throw new Error('Transcription for long recordings is disabled in settings.');
  }
  const estimatedCost = (durationSeconds ?? 0) * COST_ESTIMATES.transcriptionPerSecond;
  await enforceBudget(estimatedCost);
  const totalBytes = audioChunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const batches = totalBytes > MAX_UPLOAD_BYTES ? splitAudioIntoBatches(audioChunks, MAX_UPLOAD_BYTES) : [audioChunks];
  const apiKey = await loadApiKey();
  const transcripts = [];
  for (const [batchIndex, batch] of batches.entries()) {
    const formData = new FormData();
    formData.append('model', 'whisper-1');
    formData.append('response_format', 'json');
    if (language) {
      formData.append('language', language);
    }
    const parts = batch.map((chunk) => (chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk));
    const blob = new Blob(parts, { type: mimeType || 'audio/webm' });
    formData.append('file', blob, `audio-${batchIndex}.webm`);
    const response = await fetch(`${OPENAI_BASE_URL}/audio/transcriptions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`
      },
      body: formData
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Transcription failed: ${text}`);
    }
    const contentType = response.headers.get('content-type') ?? '';
    const data = contentType.includes('application/json') ? await response.json() : { text: await response.text() };
    transcripts.push(data.text ?? '');
  }
  await incrementUsage({ transcriptionSeconds: durationSeconds ?? 0, estimatedCost });
  return transcripts.join(' ').trim();
}

async function handleAskQuestion(message) {
  // Routes user questions through GPT with page-aware context.
  const { question, context, modelPreference } = message ?? {};
  if (!question) {
    throw new Error('Question is required');
  }
  await enforceBudget();
  const apiKey = await loadApiKey();
  const settings = await loadSettings();
  const model = selectModel(modelPreference ?? settings.preferredModel, settings.allowGpt4o);
  let questionContext = context;
  if (!questionContext) {
    const pageData = await collectPageData();
    questionContext = buildContextSnippet(pageData);
  }
  const payload = {
    model,
    messages: [
      {
        role: 'system',
        content: `You are an accessibility assistant that answers questions succinctly about the current web page. Respond in ${settings.outputLanguage ?? 'the user\'s language'} when possible.`
      },
      {
        role: 'user',
        content: `Context:\n${questionContext}\n---\nQuestion: ${question}`
      }
    ],
    temperature: 0.2
  };
  const result = await callChatCompletion(apiKey, payload, model);
  return result.choices?.[0]?.message?.content ?? '';
}

async function handleRequestPageRead(message, sender) {
  // Collects page text, optionally summarizes it, and streams TTS audio back to the popup.
  const { portId, voice, ttsFormat, targetLanguage, summarizationEnabled } = message ?? {};
  if (!portId) {
    throw new Error('Missing audio port identifier');
  }
  await enforceBudget();
  const port = audioPorts.get(portId);
  if (!port) {
    throw new Error('Audio stream is not connected');
  }
  const apiKey = await loadApiKey();
  const settings = await loadSettings();
  const controller = new AbortController();
  activeNarrations.set(portId, controller);
  try {
    if (controller.signal.aborted) {
      return { cancelled: true };
    }

    const pageData = await collectPageData();

    const summarizationRequired = Boolean((summarizationEnabled ?? settings.summarizationEnabled) && pageData.text && pageData.text.length > (settings.summarizationThreshold ?? DEFAULT_SETTINGS.summarizationThreshold));
    let narrationText = pageData.text ?? '';
    if (summarizationRequired) {
      narrationText = await getSummary({
        apiKey,
        pageData,
        targetLanguage: targetLanguage ?? settings.outputLanguage,
        signal: controller.signal
      });
    }

    if (controller.signal.aborted) {
      return { cancelled: true };
    }

    const language = targetLanguage ?? settings.outputLanguage;
    if (controller.signal.aborted) {
      return { cancelled: true };
    }
    const result = await playNarration({
      apiKey,
      port,
      voice: voice ?? settings.voice,
      format: ttsFormat ?? settings.ttsFormat,
      text: narrationText,
      pageData,
      language,
      signal: controller.signal
    });
    if (result.cancelled) {
      return { cancelled: true };
    }
    await incrementUsage({
      ttsCharacters: narrationText.length,
      estimatedCost: narrationText.length * COST_ESTIMATES.ttsPerCharacter
    });
    return { completed: true };
  } catch (error) {
    if (error?.name === 'AbortError') {
      return { cancelled: true };
    }
    throw error;
  } finally {
    activeNarrations.delete(portId);
  }
}

async function handleStopPageRead(message) {
  const { portId } = message ?? {};
  if (!portId) {
    return true;
  }
  const controller = activeNarrations.get(portId);
  if (controller) {
    controller.abort();
  }
  return true;
}

async function loadSettings() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.SETTINGS);
  return stored[STORAGE_KEYS.SETTINGS] ?? DEFAULT_SETTINGS;
}

async function loadUsage() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.USAGE);
  return stored[STORAGE_KEYS.USAGE];
}

async function incrementUsage(partial) {
  const usage = await loadUsage();
  const updated = {
    ...usage,
    transcriptionSeconds: (usage.transcriptionSeconds ?? 0) + (partial.transcriptionSeconds ?? 0),
    ttsCharacters: (usage.ttsCharacters ?? 0) + (partial.ttsCharacters ?? 0),
    chatTokens: (usage.chatTokens ?? 0) + (partial.chatTokens ?? 0),
    estimatedCost: Number(((usage.estimatedCost ?? 0) + (partial.estimatedCost ?? 0)).toFixed(4))
  };
  await browserApi.storage.local.set({ [STORAGE_KEYS.USAGE]: updated });
  return updated;
}

async function callChatCompletion(apiKey, payload, modelUsed, signal) {
  await enforceBudget();
  const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Chat completion failed: ${text}`);
  }
  const data = await response.json();
  const tokens = data.usage?.total_tokens ?? 0;
  const cost = calculateChatCost(tokens, modelUsed);
  await incrementUsage({ chatTokens: tokens, estimatedCost: cost });
  return data;
}

function calculateChatCost(tokens, model) {
  const perToken = model.includes('gpt-4o') ? COST_ESTIMATES.gpt4oPer1kTokens : COST_ESTIMATES.gpt35Per1kTokens;
  return (tokens / 1000) * perToken;
}

async function enforceBudget(extraCost = 0) {
  const settings = await loadSettings();
  const limit = Number(settings.monthlyLimit ?? 0);
  if (!limit) {
    return;
  }
  const usage = await loadUsage();
  const currentCost = Number(usage.estimatedCost ?? 0);
  if (currentCost + extraCost > limit) {
    throw new Error('Monthly budget exceeded. Adjust cost controls to continue.');
  }
}

function selectModel(preferred, allowGpt4o) {
  if (preferred && preferred.includes('gpt-4o') && !allowGpt4o) {
    return 'gpt-3.5-turbo';
  }
  return preferred ?? 'gpt-3.5-turbo';
}

async function getActiveTab() {
  const [tab] = await browserApi.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    throw new Error('No active tab detected');
  }
  return tab;
}

async function getSummary({ apiKey, pageData, targetLanguage, signal }) {
  const cache = await loadCache();
  const hash = await hashText(pageData.text);
  const cacheKey = `${pageData.url}:${targetLanguage}:${hash}`;
  if (signal?.aborted) {
    const abortError = new Error('Aborted');
    abortError.name = 'AbortError';
    throw abortError;
  }
  if (cache.summaries[cacheKey]) {
    return cache.summaries[cacheKey].text;
  }
  const model = selectModel('gpt-3.5-turbo', false);
  const basePrompt = `Summarize this page for a visually-impaired user in ${targetLanguage}. Preserve the hierarchy of headings and include critical links when relevant. Keep the response concise.`;
  const chunked = chunkText(pageData.text, 4500);
  const summaries = [];
  for (const chunk of chunked) {
    if (signal?.aborted) {
      const abortError = new Error('Aborted');
      abortError.name = 'AbortError';
      throw abortError;
    }
    const payload = {
      model,
      temperature: 0.2,
      messages: [
        { role: 'system', content: basePrompt },
        {
          role: 'user',
          content: buildSummaryPrompt({ chunk, pageData })
        }
      ]
    };
    const data = await callChatCompletion(apiKey, payload, model, signal);
    summaries.push(data.choices?.[0]?.message?.content ?? '');
  }
  const finalSummary = summaries.join('\n');
  await storeSummary(cacheKey, finalSummary);
  return finalSummary;
}

function buildSummaryPrompt({ chunk, pageData }) {
  const headings = pageData.headings?.length ? `Headings: ${pageData.headings.join(' | ')}` : '';
  const links = pageData.links?.length
    ? `Important links: ${pageData.links.slice(0, 10).map((link) => `${link.text} -> ${link.href}`).join(' | ')}`
    : '';
  return [`Title: ${pageData.title}`, headings, links, 'Content:', chunk].filter(Boolean).join('\n');
}

function chunkText(text, size) {
  const chunks = [];
  let index = 0;
  while (index < text.length) {
    chunks.push(text.slice(index, index + size));
    index += size;
  }
  return chunks;
}

async function playNarration({ apiKey, port, voice, format, text, pageData, language, signal }) {
  const cacheKey = await buildAudioCacheKey({ text, voice, format, language, url: pageData.url });
  const cache = await loadCache();
  const cachedAudio = cache.audio[cacheKey];
  if (cachedAudio) {
    if (signal?.aborted) {
      return { cancelled: true };
    }
    await streamCachedAudio(port, cachedAudio, format, signal);
    return { cancelled: Boolean(signal?.aborted) };
  }
  await enforceBudget(text.length * COST_ESTIMATES.ttsPerCharacter);
  port.postMessage({ type: 'tts-start', format });
  const payload = {
    model: 'gpt-4o-mini-tts',
    voice: voice || 'alloy',
    format: format || 'mp3',
    input_language: language,
    response_format: format || 'mp3',
    text
  };
  const chunks = [];
  try {
    const response = await fetch(`${OPENAI_BASE_URL}/audio/speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload),
      signal
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Speech synthesis failed: ${errorText}`);
    }
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Streaming is not supported in this browser');
    }
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        const chunkCopy = new Uint8Array(value);
        chunks.push(chunkCopy);
        const transferable = chunkCopy.buffer.slice(0);
        port.postMessage({ type: 'tts-chunk', chunk: transferable }, [transferable]);
      }
    }
  } catch (error) {
    if (error?.name === 'AbortError') {
      return { cancelled: true };
    }
    throw error;
  }
  port.postMessage({ type: 'tts-end' });
  const combined = combineChunks(chunks);
  await storeAudio(cacheKey, combined, format);
  return { cancelled: false };
}

async function streamCachedAudio(port, cachedAudio, format, signal) {
  if (signal?.aborted) {
    return;
  }
  port.postMessage({ type: 'tts-start', format });
  const binary = Uint8Array.from(atob(cachedAudio.data), (char) => char.charCodeAt(0));
  if (signal?.aborted) {
    return;
  }
  port.postMessage({ type: 'tts-chunk', chunk: binary.buffer }, [binary.buffer]);
  if (signal?.aborted) {
    return;
  }
  port.postMessage({ type: 'tts-end' });
  return;
}

function combineChunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const combined = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }
  return combined.buffer;
}

async function storeSummary(cacheKey, text) {
  const cache = await loadCache();
  cache.summaries[cacheKey] = { text, timestamp: Date.now() };
  enforceCacheLimit(cache.summaries);
  await browserApi.storage.local.set({ [STORAGE_KEYS.CACHE]: cache });
}

async function storeAudio(cacheKey, buffer, format) {
  const cache = await loadCache();
  cache.audio[cacheKey] = {
    data: arrayBufferToBase64(buffer),
    format,
    timestamp: Date.now()
  };
  enforceCacheLimit(cache.audio);
  await browserApi.storage.local.set({ [STORAGE_KEYS.CACHE]: cache });
}

function enforceCacheLimit(store) {
  const entries = Object.entries(store);
  if (entries.length <= MAX_CACHE_ENTRIES) {
    return;
  }
  entries
    .sort((a, b) => (a[1].timestamp ?? 0) - (b[1].timestamp ?? 0))
    .slice(0, entries.length - MAX_CACHE_ENTRIES)
    .forEach(([key]) => delete store[key]);
}

async function loadCache() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.CACHE);
  return stored[STORAGE_KEYS.CACHE] ?? { summaries: {}, audio: {} };
}

async function hashText(text) {
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(text));
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function buildAudioCacheKey({ text, voice, format, language, url }) {
  const textHash = await hashText(text);
  return `${url}:${voice}:${format}:${language}:${textHash}`;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function collectPageData() {
  // Requests structured page data from the content script to keep background-only access to sensitive tokens.
  const activeTab = await getActiveTab();
  try {
    const pageData = await browserApi.tabs.sendMessage(activeTab.id, { type: 'COLLECT_PAGE_TEXT' });
    if (pageData?.error) {
      throw new Error(pageData.error);
    }
    return pageData;
  } catch (error) {
    console.error('Failed to collect page text', error);
    throw new Error('Unable to collect page text. Ensure the page allows content scripts.');
  }
}

function buildContextSnippet(pageData, limit = 6000) {
  // Reduces page payload before sending it to the chat completion endpoint.
  if (!pageData) {
    return '';
  }
  const parts = [];
  if (pageData.title) {
    parts.push(`Title: ${pageData.title}`);
  }
  if (Array.isArray(pageData.headings) && pageData.headings.length) {
    parts.push(`Headings: ${pageData.headings.slice(0, 10).join(' | ')}`);
  }
  if (Array.isArray(pageData.links) && pageData.links.length) {
    const keyLinks = pageData.links.slice(0, 5).map((link) => `${link.text} (${link.href})`).join(' | ');
    parts.push(`Links: ${keyLinks}`);
  }
  if (pageData.text) {
    const normalizedText = pageData.text.replace(/\s+/g, ' ').trim();
    parts.push(normalizedText.slice(0, limit));
  }
  const context = parts.filter(Boolean).join('\n');
  return context.length > limit ? context.slice(0, limit) : context;
}

async function loadApiKey() {
  const stored = await browserApi.storage.local.get(STORAGE_KEYS.ENCRYPTED_KEY);
  const record = stored[STORAGE_KEYS.ENCRYPTED_KEY];
  if (!record) {
    throw new Error('Add your OpenAI API key to use the extension.');
  }
  if (!sessionState.passphrase) {
    throw new Error('Unlock the API key with your passphrase.');
  }
  return decryptSecret(record, sessionState.passphrase);
}

async function encryptSecret(secret, passphrase) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const encrypted = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoder.encode(secret));
  return {
    cipherText: arrayBufferToBase64(encrypted),
    iv: arrayBufferToBase64(iv.buffer),
    salt: arrayBufferToBase64(salt.buffer)
  };
}

async function decryptSecret(record, passphrase) {
  const iv = base64ToArrayBuffer(record.iv);
  const salt = base64ToArrayBuffer(record.salt);
  const cipher = base64ToArrayBuffer(record.cipherText);
  const key = await deriveKey(passphrase, new Uint8Array(salt));
  const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: new Uint8Array(iv) }, key, cipher);
  return decoder.decode(decrypted);
}

async function deriveKey(passphrase, salt) {
  const baseKey = await crypto.subtle.importKey('raw', encoder.encode(passphrase), { name: 'PBKDF2' }, false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 250000, hash: 'SHA-256' },
    baseKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

function base64ToArrayBuffer(value) {
  const binary = atob(value);
  const length = binary.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

// Ensures OpenAI's 25MB upload limit is not exceeded by a single transcription request.
function splitAudioIntoBatches(chunks, maxBytes) {
  const batches = [];
  let currentBatch = [];
  let currentSize = 0;
  for (const chunk of chunks) {
    if (chunk.byteLength > maxBytes) {
      throw new Error('Audio chunk exceeds the maximum allowed size of 25MB.');
    }
    if (currentSize + chunk.byteLength > maxBytes) {
      batches.push(currentBatch);
      currentBatch = [];
      currentSize = 0;
    }
    currentBatch.push(chunk);
    currentSize += chunk.byteLength;
  }
  if (currentBatch.length) {
    batches.push(currentBatch);
  }
  return batches;
}
