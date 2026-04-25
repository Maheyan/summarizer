import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/server';

dotenv.config();

const { GEMINI_API_KEY, PORT = 3000 } = process.env;
if (!GEMINI_API_KEY) { console.error('Missing GEMINI_API_KEY in .env'); process.exit(1); }

const genai       = new GoogleGenerativeAI(GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(GEMINI_API_KEY);

// ── Model registry (free-tier) ──
export const MODELS = {
  flash2: {
    id:          'gemini-2.0-flash',
    label:       'Gemini 2.0 Flash',
    description: 'Recommended — fast, free tier',
  },
  flashLite: {
    id:          'gemini-2.0-flash-lite',
    label:       'Gemini 2.0 Flash Lite',
    description: 'Lightest — best for very long documents',
  },
};

// ── Structured output schema ──
const RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    pageCount: { type: 'integer' },
    summary:   { type: 'string' },
    pages: {
      type:  'array',
      items: {
        type: 'object',
        properties: {
          page: { type: 'integer' },
          text: { type: 'string' },
        },
        required: ['page', 'text'],
      },
    },
  },
  required: ['pageCount', 'summary', 'pages'],
};

// ── Gemini: analyze uploaded PDF ──
async function analyzePdf(fileUri, modelKey, retries = 4) {
  const model = genai.getGenerativeModel(
    {
      model: MODELS[modelKey].id,
      generationConfig: {
        responseMimeType: 'application/json',
        responseSchema:   RESPONSE_SCHEMA,
      },
    },
    { apiVersion: 'v1beta' }
  );

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await model.generateContent([
        { fileData: { mimeType: 'application/pdf', fileUri } },
        'Analyze this document carefully. ' +
        'For each page extract all key content: text, data, headings, figures, and labels. ' +
        'If the document contains non-Latin scripts (Bangla, Arabic, etc.) or handwritten content, process them accurately. ' +
        'Then write a detailed, comprehensive, well-structured summary covering all key concepts, arguments, data points, and conclusions.',
      ]);
      return JSON.parse(result.response.text());
    } catch (err) {
      if (err?.message?.includes('429') && attempt < retries) {
        const delay = Math.min(1000 * 2 ** attempt, 30000);
        console.warn(`  ⚠ Rate limited — retrying in ${delay / 1000}s (attempt ${attempt + 1})`);
        await new Promise((r) => setTimeout(r, delay));
      } else {
        throw err;
      }
    }
  }
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

  const modelKey  = MODELS[req.body?.model] ? req.body.model : 'flash2';
  const pdfPath   = req.file.path;
  const pdfName   = req.file.originalname;
  console.log(`\n[${new Date().toISOString()}] Processing ${pdfName} [${MODELS[modelKey].label}]`);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  let fileUri = null;
  try {
    // 1. Upload PDF to Gemini Files API
    send('uploading', {});
    const uploaded = await fileManager.uploadFile(pdfPath, {
      mimeType:    'application/pdf',
      displayName: pdfName,
    });
    fileUri = uploaded.file.uri;
    console.log(`  ✓ Uploaded: ${fileUri}`);

    // 2. Analyze (one call — handles text, OCR, Bangla, handwriting)
    send('analyzing', {});
    const result = await analyzePdf(fileUri, modelKey);
    console.log(`  ✓ Done — ${result.pageCount} pages, summary: ${result.summary.length} chars`);

    send('done', { pageCount: result.pageCount, pages: result.pages, summary: result.summary });
  } catch (err) {
    console.error('  ✗ Error:', err.message);
    send('error', { message: err.message });
  } finally {
    res.end();
    fs.unlink(pdfPath).catch(() => {});
    if (fileUri) fileManager.deleteFile(fileUri).catch(() => {});
  }
});

app.use((err, _req, res, _next) => {
  res.status(400).json({ error: err.message });
});

app.listen(PORT, () => console.log(`📚 LectureLens running on http://localhost:${PORT}`));
