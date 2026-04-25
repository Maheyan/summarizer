import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import dotenv from 'dotenv';
import { createRequire } from 'module';

const _require = createRequire(import.meta.url);

// process.getBuiltinModule polyfill for Node.js < 21.2.0
if (typeof process.getBuiltinModule !== 'function') {
  process.getBuiltinModule = (id) => {
    const name = id.startsWith('node:') ? id.slice(5) : id;
    if (name === 'require') return _require;
    try { return _require(name); } catch { return undefined; }
  };
}

const { getDocument, GlobalWorkerOptions } = await import('pdfjs-dist/legacy/build/pdf.mjs');
GlobalWorkerOptions.workerSrc = `file://${_require.resolve('pdfjs-dist/legacy/build/pdf.worker.mjs')}`;

dotenv.config();

const { CF_ACCOUNT_ID, CF_API_TOKEN, PORT = 3000 } = process.env;

if (!CF_ACCOUNT_ID || !CF_API_TOKEN) {
  console.error('Missing CF_ACCOUNT_ID or CF_API_TOKEN in .env');
  process.exit(1);
}

const CF_BASE = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/ai/run`;

const app = express();
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === 'application/pdf') cb(null, true);
    else cb(new Error('Only PDF files allowed'));
  },
});

// ----- Cloudflare AI helper -----
async function cfRun(model, payload) {
  const res = await fetch(`${CF_BASE}/${model}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${CF_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.success === false) {
    const errMsg = data.errors?.map((e) => e.message).join('; ') || res.statusText;
    throw new Error(`CF AI [${model}] failed: ${errMsg}`);
  }
  return data.result;
}

// ----- BART: hierarchical summarization -----
const CHUNK_CHARS = 3500;

async function summarizeChunk(text, { maxLen = 400, minLen = 120 } = {}) {
  const result = await cfRun('@cf/facebook/bart-large-cnn', {
    input_text: text,
    max_length: maxLen,
    min_length: minLen,
  });
  return result?.summary?.trim() || '';
}

async function summarizeText(text) {
  if (!text.trim()) return '';
  if (text.length <= CHUNK_CHARS) return summarizeChunk(text, { maxLen: 600, minLen: 200 });

  const chunks = [];
  let buf = '';
  for (const para of text.split(/\n\n+/)) {
    if ((buf + '\n\n' + para).length > CHUNK_CHARS && buf) {
      chunks.push(buf);
      buf = para;
    } else {
      buf = buf ? `${buf}\n\n${para}` : para;
    }
  }
  if (buf) chunks.push(buf);

  const partials = [];
  for (const chunk of chunks) partials.push(await summarizeChunk(chunk, { maxLen: 400, minLen: 120 }));

  const combined = partials.join('\n\n');
  if (combined.length <= CHUNK_CHARS) return summarizeChunk(combined, { maxLen: 700, minLen: 250 });
  return summarizeText(combined);
}

// ----- PDF text extraction -----
async function extractPageText(pdfDoc, pageNum) {
  const page    = await pdfDoc.getPage(pageNum);
  const content = await page.getTextContent();
  return content.items.map((item) => item.str).join(' ').trim();
}

// ----- Routes -----
app.use(express.static('.'));

app.post('/api/summarize', upload.single('pdf'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No PDF uploaded' });

  const pdfPath = req.file.path;
  console.log(`\n[${new Date().toISOString()}] Processing ${req.file.originalname}`);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const pdfData = new Uint8Array(await fs.readFile(pdfPath));
    const pdfDoc  = await getDocument({ data: pdfData }).promise;
    const pageCount = pdfDoc.numPages;
    send('pdf_loaded', { pageCount });

    const pageTexts = [];
    for (let i = 1; i <= pageCount; i++) {
      const text = await extractPageText(pdfDoc, i);
      pageTexts.push({ page: i, text });
      console.log(`  ✓ Page ${i}/${pageCount} extracted (${text.length} chars)`);
      send('page_done', { page: i, pageCount, text });
    }

    send('summarizing', {});
    const fullText = pageTexts.map((p) => `--- Page ${p.page} ---\n${p.text}`).join('\n\n');
    const summary  = await summarizeText(fullText);

    send('done', { pageCount, pages: pageTexts, fullText, summary });
    console.log(`  ✓ Done — summary: ${summary.length} chars`);
  } catch (err) {
    console.error('  ✗ Error:', err.message);
    send('error', { message: err.message });
  } finally {
    res.end();
    fs.unlink(pdfPath).catch(() => {});
  }
});

app.use((err, _req, res, _next) => {
  res.status(400).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`📚 Class PDF Summarizer running on http://localhost:${PORT}`);
});
