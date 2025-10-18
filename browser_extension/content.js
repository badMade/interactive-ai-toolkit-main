// Content script extracts visible text and executes basic page controls on demand.
const browserApi = typeof browser !== 'undefined' ? browser : chrome;

const INVISIBLE_DISPLAY = new Set(['none', 'hidden']);
const INVISIBLE_VISIBILITY = new Set(['hidden', 'collapse']);

function isElementVisible(element) {
  if (!element || element.nodeType !== Node.ELEMENT_NODE) {
    return false;
  }
  const style = window.getComputedStyle(element);
  if (INVISIBLE_DISPLAY.has(style.display) || INVISIBLE_VISIBILITY.has(style.visibility) || Number(style.opacity) === 0) {
    return false;
  }
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function extractVisibleText(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.parentElement) {
        return NodeFilter.FILTER_REJECT;
      }
      const trimmed = node.textContent.replace(/\s+/g, ' ').trim();
      if (!trimmed) {
        return NodeFilter.FILTER_REJECT;
      }
      return isElementVisible(node.parentElement) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    }
  });

  const chunks = [];
  while (walker.nextNode()) {
    chunks.push(walker.currentNode.textContent.replace(/\s+/g, ' ').trim());
  }
  return chunks.join('\n');
}

function collectHeadings(root) {
  const headings = Array.from(root.querySelectorAll('h1, h2, h3, h4, h5, h6'))
    .filter(isElementVisible)
    .map((heading) => heading.textContent.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  return headings;
}

function collectImportantLinks(root, limit = 20) {
  const links = [];
  for (const anchor of root.querySelectorAll('a[href]')) {
    if (!isElementVisible(anchor)) {
      continue;
    }
    const text = anchor.textContent.replace(/\s+/g, ' ').trim();
    if (!text) {
      continue;
    }
    links.push({ text, href: anchor.href });
    if (links.length >= limit) {
      break;
    }
  }
  return links;
}

async function handleMessage(message) {
  switch (message?.type) {
    case 'COLLECT_PAGE_TEXT': {
      const body = document.body || document.documentElement;
      const text = extractVisibleText(body);
      const headings = collectHeadings(body);
      const links = collectImportantLinks(body);
      return {
        url: window.location.href,
        title: document.title,
        text,
        headings,
        links,
        language: document.documentElement.lang || navigator.language
      };
    }
    case 'CONTROL_ACTION': {
      const { command } = message;
      if (command?.action === 'scroll') {
        const delta = command.direction === 'up' ? -window.innerHeight : window.innerHeight;
        window.scrollBy({ top: delta, behavior: 'smooth' });
      }
      return { handled: true };
    }
    default:
      return undefined;
  }
}

browserApi.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const maybePromise = handleMessage(message);
  if (maybePromise && typeof maybePromise.then === 'function') {
    maybePromise.then(sendResponse).catch((error) => {
      console.error('Content script error', error);
      sendResponse({ error: error?.message ?? 'Failed to collect page text' });
    });
    return true;
  }
  if (maybePromise !== undefined) {
    sendResponse(maybePromise);
  }
  return false;
});
