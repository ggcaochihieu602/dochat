# Dochat - Trợ lý Telegram đọc tài liệu

Bot Telegram đọc văn bản, PDF (tối đa 2 trang) hoặc ảnh, tóm tắt bằng AI rồi trả lời bằng giọng nói MP3 tiếng Việt. Dành riêng cho gia đình, chạy gần như trong free tier.

## Kiến trúc

```text
Telegram
  -> Cloudflare Worker (webhook, whitelist, state, job queue)
      -> Render FastAPI (extract, OCR, summary, Azure TTS)
          -> Telegram (MP3 / text)
```

- Worker nhận webhook, trả lời ngay, đưa job vào Queue (chống Render sleep).
- Backend Render xử lý nội dung và gửi kết quả về Telegram.
- `SUMMARY_PROMPT.md` đọc tại runtime - ngài chỉnh sửa prompt tóm tắt tại file này, không cần sửa code.

## Cấu trúc

```text
backend/          FastAPI + extraction + OCR + summary + Azure TTS
worker/           Cloudflare Worker (wrangler + TypeScript)
Dockerfile        Backend image cho Render
SUMMARY_PROMPT.md Prompt tóm tắt (có thể chỉnh)
.env.example      Mẫu biến môi trường
```

## Yêu cầu

- Tài khoản Cloudflare (Workers + KV + Queue + Workers AI)
- Tài khoản Azure (Speech, free F0 500.000 ký tự/tháng)
- Tài khoản Render (Web Service free)
- Telegram bot token từ @BotFather

## Triển khai

### 1. Backend (Render)

1. Tạo repo trên GitHub và đẩy toàn bộ project (đã có `.gitignore` chặn file secret).
2. Trên Render: New > Web Service > chọn repo, instance Free.
3. Trong Render Settings đặt các biến môi trường:

```text
TELEGRAM_BOT_TOKEN=
TELEGRAM_WEBHOOK_SECRET=
INTERNAL_SECRET=
ALLOWED_USER_IDS=851987991
BACKEND_URL=https://your-service.onrender.com
SUMMARY_PROVIDER_URL=https://your-worker.workers.dev/internal/summarize
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=southeastasia
AZURE_VOICE_NAME=vi-VN-HoaiMyNeural
```

Render sẽ dùng Dockerfile có sẵn. URL backend có dạng `https://ten-service.onrender.com`.

### 2. Cloudflare Worker

Trong thư mục `worker`:

```bash
npm install
wrangler login
wrangler kv namespace create CONVERSATIONS   # ghi lại ID
wrangler queue create dochat-jobs
```

Cập nhật `worker/wrangler.toml`:

- Thay `REPLACE_WITH_KV_NAMESPACE_ID` bằng ID vừa tạo.
- Thay `BACKEND_URL` bằng URL Render thật.
- Đặt `ALLOWED_USER_IDS` (dấu phẩy ngăn cách nếu nhiều người).

Đặt secret cho Worker:

```bash
cd worker
wrangler secret put TELEGRAM_BOT_TOKEN
wrangler secret put TELEGRAM_WEBHOOK_SECRET
wrangler secret put INTERNAL_SECRET
wrangler deploy
```

### 3. Cấu hình Telegram webhook

```bash
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://dochat-telegram.<account>.workers.dev/webhook&secret_token=<TELEGRAM_WEBHOOK_SECRET>"
```

`TELEGRAM_WEBHOOK_SECRET` ở Worker và Render phải giống nhau (dùng chung để backend xác thực).

### 4. Whitelist

- Mặc định whitelist `851987991` (ngài).
- Thêm ID của bố bằng cách thêm vào `ALLOWED_USER_IDS` trong `wrangler.toml` (ví dụ `"851987991,123456789"`), rồi `wrangler deploy`.
- Lấy user ID Telegram của bố bằng bot `@userinfobot`.

## Chạy local (backend)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Cài Tesseract (bản Windows) có gói `vie` và `eng`, đặt đường dẫn `pytesseract.pytesseract.tesseract_cmd`. Tạo `.env` từ `.env.example`, sau đó:

```bash
uvicorn backend.main:app --reload
```

## Kiểm tra

```bash
python -m pytest backend/test_helpers.py
cd worker && npm test && npx tsc --noEmit
```

## Lưu ý bảo mật

- Không commit token, key hoặc file `.txt` secret.
- Worker chỉ nhận webhook có `X-Telegram-Bot-Api-Secret-Token` đúng.
- Backend `/process` và Worker `/internal/summarize` yêu cầu `X-Internal-Secret`.
- Không lưu tài liệu, MP3 sau khi xử lý.