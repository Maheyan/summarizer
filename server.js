import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import dotenv from 'dotenv';
import { createRequire } from 'module';
import napiCanvas from '@napi-rs/canvas';
import { GoogleGenerativeAI } from '@google/generative-ai';

const { createCanvas, DOMMatrix, ImageData, Path2D, Image } = napiCanvas;

// ── process.getBuiltinModule polyfill (Node.js < 21.2.0) ──
const _require = createRequire(import.meta.url);
if (typeof process.getBuiltinModule !== 'function') {
  process.getBuiltinModule = (id) => {
    const name = id.startsWith('node:') ? id.slice(5) : id;
    if (name === 'require') return _require;
    try { return _require(name); } catch { return undefined; }
  };
}

// ── Browser globals pdfjs-dist needs in Node.js ──
globalThis.DOMMatrix ??= DOMMatrix;
globalThis.ImageData ??= ImageData;
globalThis.Path2D    ??= Path2D;
globalThis.Image     ??= Image;

const { getDocument, GlobalWorkerOptions } = await import('pdfjs-dist/legacy/build/pdf.mjs');
GlobalWorkerOptions.workerSrc = `file://${_require.resolve('pdfjs-dist/legacy/build/pdf.worker.mjs')}`;

dotenv.config();

const { GEMINI_API_KEY, PORT = 3000 } = process.env;
if (!GEMINI_API_KEY) {
  console.error('Missing GEMINI_API_KEY in .env');
  process.exit(1);
}

const genai = new GoogleGenerativeAI(GEMINI_API_KEY);

// ── Model registry (free-tier only) ──
export const MODELS = {
  flash2: {
    id: 'gemini-2.0-flash',
    label: 'Gemini 2.0 Flash',
    description: 'Recommended — fast, multimodal, free tier',
  },
  flashLite: {
    id: 'gemini-2.0-flash-lite',
    label: 'Gemini 2.0 Flash Lite',
    description: 'Lightest — best for very long documents',
  },
};

// ── Gemini call with retry on 429 ──
async function geminiGenerate(modelKey, parts, retries = 4) {
  const model = genai.getGenerativeModel(
    { model: MODELS[modelKey].id },
    { apiVersion: 'v1' }
  );
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await model.generateContent(parts);
      return result.response.text().trim();
    } catch (err) {
      const is429 = err?.message?.includes('429') || err?.status === 429;
      if (is429 && attempt < retries) {
        const delay = Math.min(1000 * 2 ** attempt, 30000);
        console.warn(`  ⚠ Rate limited — retrying in ${delay / 1000}s (attempt ${attempt + 1}/${retries})`);
        await new Promise((r) => setTimeout(r, delay));
      } else {
        throw err;
      }
    }
  }
}

// ── Gemini: summarize text ──
async function summarizeText(modelKey, text) {
  if (!text.trim()) return '';
  return geminiGenerate(modelKey,
    'You are an expert academic summarizer. Provide a detailed, comprehensive, well-structured summary ' +
    'of the following document. Cover all key concepts, arguments, data points, and conclusions. ' +
    'Use clear prose organized by topic.\n\n' + text
  );
}

// ── Gemini Vision: OCR a page image ──
async function ocrPageWithVision(pngBuffer, modelKey) {
  return geminiGenerate(modelKey, [
    {
      inlineData: {
        data: pngBuffer.toString('base64'),
        mimeType: 'image/png',
      },
    },
    'Transcribe ALL text visible on this page exactly as written. ' +
    'If the text is handwritten, transcribe it faithfully. ' +
    'If the text is in Bangla, Arabic, or any non-Latin script, reproduce it accurately. ' +
    'Do not summarize — transcribe only.',
  ]);
}

// ── PDF page rendering (for vision OCR fallback) ──
class NodeCanvasFactory {
  create(w, h)   { const c = createCanvas(w, h); return { canvas: c, context: c.getContext('2d') }; }
  reset(d, w, h) { d.canvas.width = w; d.canvas.height = h; }
  destroy(d)     { d.canvas.width = 0; d.canvas.height = 0; }
}

async function renderPageToPng(pdfDoc, pageNum) {
  const page     = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale: 2.0 });
  const factory  = new NodeCanvasFactory();
  const data     = factory.create(viewport.width, viewport.height);
  await page.render({ canvasContext: data.context, viewport, canvasFactory: factory }).promise;
  const buf = data.canvas.toBuffer('image/png');
  factory.destroy(data);
  return buf;
}

// ── Text extraction: text layer → vision OCR fallback ──
const SPARSE_THRESHOLD = 50;

async function extractPageText(pdfDoc, pageNum, modelKey) {
  const page    = await pdfDoc.getPage(pageNum);
  const content = await page.getTextContent();
  const text    = content.items.map((i) => i.str).join(' ').trim();

  if (text.length >= SPARSE_THRESHOLD) return text;

  console.log(`  → Page ${pageNum}: sparse text (${text.length} chars), using vision OCR`);
  const png = await renderPageToPng(pdfDoc, pageNum);
  return ocrPageWithVision(png, modelKey);
}

// ── Express ──
const app = express();
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    file.mimetype === 'application/pdf' ? cb(null, true) : cb(new Error('Only PDF files allowed'));
  },
});

app.use(express.static('.'));

app.get('/api/models', (_req, res) => {
  res.json(Object.entries(MODELS).map(([key, m]) => ({ key, label: m.label, description: m.description })));
});

app.post('/api/summarize', upload.single('pdf'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No PDF uploaded' });

  const modelKey = MODELS[req.body?.model] ? req.body.model : 'flash2';
  const pdfPath  = req.file.path;
  console.log(`\n[${new Date().toISOString()}] Processing ${req.file.originalname} [${MODELS[modelKey].label}]`);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const pdfData   = new Uint8Array(await fs.readFile(pdfPath));
    const pdfDoc    = await getDocument({ data: pdfData }).promise;
    const pageCount = pdfDoc.numPages;
    send('pdf_loaded', { pageCount });

    const pageTexts = [];
    for (let i = 1; i <= pageCount; i++) {
      const text = await extractPageText(pdfDoc, i, modelKey);
      pageTexts.push({ page: i, text });
      console.log(`  ✓ Page ${i}/${pageCount} extracted (${text.length} chars)`);
      send('page_done', { page: i, pageCount, text });
    }

    send('summarizing', {});
    const fullText = pageTexts.map((p) => `--- Page ${p.page} ---\n${p.text}`).join('\n\n');
    const summary  = await summarizeText(modelKey, fullText);

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

app.listen(PORT, () => console.log(`📚 LectureLens running on http://localhost:${PORT}`));
