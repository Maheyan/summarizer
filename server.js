import express from 'express';
import multer from 'multer';
import canvasPkg from 'canvas';
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';

// pdfjs-dist v4 needs these browser globals; provide them before the module loads
const { DOMMatrix, ImageData, Path2D } = canvasPkg;
globalThis.DOMMatrix ??= DOMMatrix;
globalThis.ImageData ??= ImageData;
globalThis.Path2D ??= Path2D;

// Dynamic import so pdfjs-dist is evaluated after globals are set above
const { pdfToPng } = await import('pdf-to-png-converter');

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
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB cap
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

// ----- LLaVA: vision -> text per page -----
async function extractFromPage(pngBuffer) {
  const imageArray = Array.from(new Uint8Array(pngBuffer));
  const result = await cfRun('@cf/llava-hf/llava-1.5-7b-hf', {
    image: imageArray,
    prompt:
      'Transcribe all text from this slide or page exactly as it appears. ' +
      'Include headings, bullet points, captions, and any text in diagrams or figures. ' +
      'Briefly describe non-text visual content (charts, diagrams, images) in one sentence each. ' +
      'Preserve the logical reading order.',
    max_tokens: 768,
  });
  return result?.description?.trim() || '';
}

// ----- BART: summarization with hierarchical chunking -----
// BART caps around 1024 input tokens (~3500-4000 chars). We chunk and re-summarize.
const CHUNK_CHARS = 3500;

async function summarizeChunk(text, maxLen = 250) {
  const result = await cfRun('@cf/facebook/bart-large-cnn', {
    input_text: text,
    max_length: maxLen,
  });
  return result?.summary?.trim() || '';
}

async function summarizeText(text) {
  if (!text.trim()) return '';

  if (text.length <= CHUNK_CHARS) {
    return summarizeChunk(text, 300);
  }

  // Split into chunks at paragraph boundaries when possible
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

  // First pass: summarize each chunk
  const partials = [];
  for (const chunk of chunks) {
    partials.push(await summarizeChunk(chunk, 200));
  }

  // Second pass: combine partials and summarize again (recursively if still too long)
  const combined = partials.join('\n\n');
  if (combined.length <= CHUNK_CHARS) {
    return summarizeChunk(combined, 400);
  }
  return summarizeText(combined); // recurse for very large docs
}

// ----- Main pipeline -----
async function processPdf(pdfPath) {
  const pages = await pdfToPng(pdfPath, {
    viewportScale: 2.0, // 2x for clearer text recognition
    outputFolder: undefined, // keep in memory
  });

  const pageTexts = [];
  for (let i = 0; i < pages.length; i++) {
    const text = await extractFromPage(pages[i].content);
    pageTexts.push({ page: i + 1, text });
    console.log(`  ✓ Page ${i + 1}/${pages.length} extracted (${text.length} chars)`);
  }

  const fullText = pageTexts
    .map((p) => `--- Page ${p.page} ---\n${p.text}`)
    .join('\n\n');

  console.log(`  Summarizing ${fullText.length} chars...`);
  const summary = await summarizeText(fullText);

  return {
    pageCount: pages.length,
    pages: pageTexts,
    fullText,
    summary,
  };
}

// ----- Routes -----
app.use(express.static('.'));

app.post('/api/summarize', upload.single('pdf'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No PDF uploaded' });

  const pdfPath = req.file.path;
  console.log(`\n[${new Date().toISOString()}] Processing ${req.file.originalname}`);

  try {
    const result = await processPdf(pdfPath);
    res.json(result);
    console.log(`  ✓ Done — summary: ${result.summary.length} chars`);
  } catch (err) {
    console.error('  ✗ Error:', err.message);
    res.status(500).json({ error: err.message });
  } finally {
    fs.unlink(pdfPath).catch(() => {});
  }
});

app.use((err, _req, res, _next) => {
  res.status(400).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`📚 Class PDF Summarizer running on http://localhost:${PORT}`);
});