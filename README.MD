# Class PDF Summarizer

Upload a class PDF, get a summary. Uses Cloudflare Workers AI:

- **`@cf/llava-hf/llava-1.5-7b-hf`** — vision model that reads each page (text + diagrams)
- **`@cf/facebook/bart-large-cnn`** — abstractive summarizer

## Setup

```bash
npm install
cp .env.example .env
# Edit .env and add your Cloudflare account ID + API token
npm start
```

Open http://localhost:3000 and drop in a PDF.

## Getting Cloudflare credentials

1. **Account ID** — visible in the right sidebar of your Cloudflare dashboard.
2. **API token** — go to https://dash.cloudflare.com/profile/api-tokens → *Create Token* → use the **Workers AI** template (or custom token with `Account → Workers AI → Read`).

## How it works

```
PDF → pdf-to-png-converter (each page → PNG)
    → LLaVA (per page: extract text + describe diagrams)
    → concatenate page texts
    → BART (chunked + hierarchical summarization)
    → final summary
```

### Why hierarchical summarization?

BART-large-CNN has roughly a 1024-token input limit (~3500 chars). For multi-page lectures, the pipeline:

1. Splits combined text into chunks at paragraph boundaries.
2. Summarizes each chunk individually.
3. Concatenates the partial summaries and runs BART again on the result.
4. Recurses if the combined partial summaries are still too long.

This handles arbitrarily long PDFs.

## API

`POST /api/summarize` — multipart form with field `pdf`

**Response:**
```json
{
  "pageCount": 24,
  "pages": [{ "page": 1, "text": "..." }],
  "fullText": "...",
  "summary": "..."
}
```

## Notes & tradeoffs

- **LLaVA is overkill for text-only PDFs.** If your PDFs are mostly selectable text (not scanned slides with diagrams), running `pdf-parse` directly is much faster and cheaper. LLaVA shines on lecture slides with charts, equations rendered as images, or scanned material. You could add a hybrid path: try `pdf-parse` first, fall back to LLaVA only if the extracted text is sparse.
- **Cost scaling.** Cost grows linearly with page count (1 LLaVA call per page) plus a few BART calls. For a 30-page deck expect ~30 vision calls + ~5–10 summarization calls.
- **Rate limits.** Workers AI has per-account concurrency limits. For large batches, add a small delay or a concurrency limiter (e.g. `p-limit`) around the `extractFromPage` loop.
- **Render scale.** `viewportScale: 2.0` gives clearer text recognition. Drop to 1.5 if pages are simple to save processing time.

## File structure

```
class-pdf-summarizer/
├── server.js            # Express backend + pipeline
├── public/index.html    # Upload UI
├── package.json
├── .env.example
└── README.md
```