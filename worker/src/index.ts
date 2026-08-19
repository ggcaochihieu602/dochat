export interface Env {
  TELEGRAM_BOT_TOKEN: string;
  TELEGRAM_WEBHOOK_SECRET: string;
  INTERNAL_SECRET: string;
  BACKEND_URL: string;
  ALLOWED_USER_IDS: string;
  AI_MODEL?: string;
  CONVERSATIONS: KVNamespace;
  JOBS: Queue<Job>;
  AI: Ai;
}

type Mode = "original_audio" | "summary_text" | "summary_audio";
type SourceType = "direct_text" | "text_file" | "pdf" | "image";
type Conversation = { step: "waiting_content"; mode: Mode };
type Job = {
  update_id: number;
  chat_id: number;
  user_id: number;
  source_type: SourceType;
  mode: Mode;
  file_id?: string;
  text?: string;
  status_message_id?: number;
  approved?: boolean;
  parent_update_id?: number;
};

const DENIED_MESSAGE = "Bot này chỉ dành cho người được cấp quyền";
const STATE_TTL_SECONDS = 3600;
const JOB_TTL_SECONDS = 86400;
const MAX_FILE_BYTES = 15 * 1024 * 1024;

const NORMALIZE_PROMPT = `Ngài là trợ lý chuẩn hóa văn bản tiếng Việt để đọc thành giọng nói. Đây là văn bản lấy từ OCR hoặc tài liệu, các dòng có thể bị ngắt tùy ý, thiếu dấu câu cuối câu, và có thể là văn bản hành chính 2 cột bị trộn lẫn.

Nếu văn bản là văn bản hành chính Việt Nam, hãy nhận diện và tách rõ các thành phần sau, mỗi thành phần là một dòng/đoạn riêng, theo đúng trình tự:
1. Tên cơ quan ban hành (ví dụ: CỤC KHÍ TƯỢNG THỦY VĂN, TRUNG TÂM DỰ BÁO...)
2. Quốc hiệu và tiêu ngữ: "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" và "Độc lập - Tự do - Hạnh phúc"
3. Số, ký hiệu văn bản (ví dụ: Số: 123/QĐ-XXX)
4. Địa danh, ngày tháng năm (ví dụ: Hà Nội, ngày 15 tháng 8 năm 2026)
5. Tiêu đề văn bản
6. Nội dung chính

Nhiệm vụ chung:
- Xác định ranh giới câu: nếu dòng tiếp theo vẫn là một phần của câu hiện tại thì nối liền bằng dấu cách; nếu câu đã kết thúc thì thêm dấu chấm (.), dấu chấm hỏi (?) hoặc dấu chấm than (!) cho phù hợp.
- Thêm dấu phẩy hoặc chấm phẩy ở những chỗ cần ngắt nghỉ tự nhiên mà văn bản gốc còn thiếu.
- Dùng dấu xuống dòng để tách các đoạn lớn.
- Giữ NGUYÊN VẸN toàn bộ nội dung: không viết lại, không tóm tắt, không thêm bớt từ ngữ, không thay đổi số liệu, tên riêng hay chữ viết tắt.
- Vì ảnh chụp 2 cột có thể làm các thành phần tiêu đề bị lẫn thứ tự, ngài CÓ THỂ sắp xếp lại các thành phần tiêu đề (tên cơ quan, quốc hiệu - tiêu ngữ, số ký hiệu, địa danh ngày tháng, tiêu đề) cho đúng trình tự quy chuẩn. Phần nội dung chính phải giữ đúng thứ tự.
- Xuất ra DUY NHẤT văn bản đã chuẩn hóa, không kèm lời giải thích, không kèm lời dẫn.

Văn bản:
${"{{DOCUMENT_TEXT}}"}`;

export function normalizePrompt(text: string): string {
  return NORMALIZE_PROMPT.replace("{{DOCUMENT_TEXT}}", text);
}

export function parseAllowedUserIds(value: string): Set<number> {
  return new Set(
    value
      .split(",")
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isSafeInteger(item) && item > 0),
  );
}

function isAllowed(userId: number, env: Env): boolean {
  return parseAllowedUserIds(env.ALLOWED_USER_IDS).has(userId);
}

function apiUrl(env: Env, method: string): string {
  return `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`;
}

async function telegram<T = Record<string, unknown>>(
  env: Env,
  method: string,
  body: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(apiUrl(env, method), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as { ok: boolean; result?: T };
  if (!response.ok || !payload.ok || payload.result === undefined) {
    throw new Error(`Telegram method failed: ${method}`);
  }
  return payload.result;
}

const homeKeyboard = {
  inline_keyboard: [
    [{ text: "Đọc tài liệu", callback_data: "action:read" }],
    [{ text: "Tóm tắt tài liệu", callback_data: "action:summary" }],
  ],
};

async function showHome(env: Env, chatId: number): Promise<void> {
  await telegram(env, "sendMessage", {
    chat_id: chatId,
    text: "Trợ lý Dochat có thể giúp gì cho ngài?",
    reply_markup: homeKeyboard,
  });
}

async function setMode(env: Env, userId: number, chatId: number, mode: Mode): Promise<void> {
  const state: Conversation = { step: "waiting_content", mode };
  await env.CONVERSATIONS.put(`state:${userId}`, JSON.stringify(state), {
    expirationTtl: STATE_TTL_SECONDS,
  });
  await telegram(env, "sendMessage", {
    chat_id: chatId,
    text: "Ngài gửi văn bản, PDF tối đa 2 trang hoặc một ảnh cho tôi.",
  });
}

async function handleCallback(update: any, env: Env): Promise<void> {
  const query = update.callback_query;
  const chatId = query.message?.chat?.id;
  if (!chatId) return;

  if (!isAllowed(query.from.id, env)) {
    await telegram(env, "answerCallbackQuery", { callback_query_id: query.id });
    await telegram(env, "sendMessage", { chat_id: chatId, text: DENIED_MESSAGE });
    return;
  }
  await telegram(env, "answerCallbackQuery", { callback_query_id: query.id });

  const jobMatch = /^job:(approve|reject):(\d+):(\w+)$/.exec(query.data);
  if (jobMatch) {
    const action = jobMatch[1];
    const parentUpdateId = Number(jobMatch[2]);
    const mode = jobMatch[3] as Mode;
    if (action === "approve") {
      const job: Job = {
        update_id: Date.now(),
        parent_update_id: parentUpdateId,
        chat_id: chatId,
        user_id: query.from.id,
        source_type: "direct_text",
        mode,
        approved: true,
      };
      await env.JOBS.send(job);
      await telegram(env, "answerCallbackQuery", {
        callback_query_id: query.id,
        text: "Đã nhận. Trợ lý Dochat đang tạo giọng nói...",
      });
    } else {
      await telegram(env, "answerCallbackQuery", {
        callback_query_id: query.id,
        text: "Đã hủy.",
      });
    }
    try {
      await telegram(env, "editMessageReplyMarkup", {
        chat_id: chatId,
        message_id: query.message.message_id,
        reply_markup: { inline_keyboard: [] },
      });
    } catch {
      // Already acknowledged; ignore.
    }
    return;
  }

  switch (query.data) {
    case "action:read":
      await telegram(env, "sendMessage", {
        chat_id: chatId,
        text: "Ngài muốn tôi đọc thế nào?",
        reply_markup: {
          inline_keyboard: [
            [{ text: "Đọc nguyên nội dung", callback_data: "mode:original_audio" }],
            [{ text: "Tóm tắt rồi đọc", callback_data: "mode:summary_audio" }],
            [{ text: "Về menu chính", callback_data: "action:home" }],
          ],
        },
      });
      return;
    case "action:summary":
      await telegram(env, "sendMessage", {
        chat_id: chatId,
        text: "Ngài muốn nhận kết quả thế nào?",
        reply_markup: {
          inline_keyboard: [
            [{ text: "Chỉ nhận bản tóm tắt", callback_data: "mode:summary_text" }],
            [{ text: "Tóm tắt và MP3", callback_data: "mode:summary_audio" }],
            [{ text: "Về menu chính", callback_data: "action:home" }],
          ],
        },
      });
      return;
    case "action:home":
      await env.CONVERSATIONS.delete(`state:${query.from.id}`);
      await showHome(env, chatId);
      return;
    case "mode:original_audio":
      await setMode(env, query.from.id, chatId, "original_audio");
      return;
    case "mode:summary_text":
      await setMode(env, query.from.id, chatId, "summary_text");
      return;
    case "mode:summary_audio":
      await setMode(env, query.from.id, chatId, "summary_audio");
      return;
  }
}

function commandOf(text: string): string {
  return text.trim().split(/\s+/, 1)[0].split("@", 1)[0].toLowerCase();
}

function sourceFromMessage(message: any): {
  source_type: SourceType;
  file_id?: string;
  text?: string;
  file_size?: number;
} | null {
  if (typeof message.text === "string" && !message.text.startsWith("/")) {
    return { source_type: "direct_text", text: message.text };
  }
  const photo = message.photo?.at(-1);
  if (photo) {
    return { source_type: "image", file_id: photo.file_id, file_size: photo.file_size };
  }
  const document = message.document;
  if (!document) return null;

  const mime = String(document.mime_type || "").toLowerCase();
  const name = String(document.file_name || "").toLowerCase();
  if (mime === "application/pdf" || name.endsWith(".pdf")) {
    return { source_type: "pdf", file_id: document.file_id, file_size: document.file_size };
  }
  if (mime.startsWith("image/") || /\.(jpe?g|png|webp)$/.test(name)) {
    return { source_type: "image", file_id: document.file_id, file_size: document.file_size };
  }
  if (mime === "text/plain" || /\.(txt|md)$/.test(name)) {
    return { source_type: "text_file", file_id: document.file_id, file_size: document.file_size };
  }
  return null;
}

async function handleMessage(update: any, env: Env): Promise<void> {
  const message = update.message;
  if (!message?.from || !message.chat) return;
  const userId = message.from.id as number;
  const chatId = message.chat.id as number;
  console.log("message_from", { userId, chatId, allowed: isAllowed(userId, env), type: message.chat.type });
  const command =
    typeof message.text === "string" && message.text.trim().startsWith("/")
      ? commandOf(message.text)
      : "";
  if (command === "/myid") {
    await telegram(env, "sendMessage", { chat_id: chatId, text: `ID Telegram của ngài là: ${userId}` });
    return;
  }

  if (!isAllowed(userId, env)) {
    await telegram(env, "sendMessage", { chat_id: chatId, text: DENIED_MESSAGE });
    return;
  }
  if (message.chat.type !== "private") {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Ngài vui lòng nhắn riêng với Trợ lý Dochat để bảo vệ nội dung tài liệu.",
    });
    return;
  }

  if (command === "/start") {
    await env.CONVERSATIONS.delete(`state:${userId}`);
    await showHome(env, chatId);
    return;
  }
  if (command === "/cancel") {
    await env.CONVERSATIONS.delete(`state:${userId}`);
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Đã hủy. Ngài có thể chọn lại từ đầu.",
      reply_markup: homeKeyboard,
    });
    return;
  }
  if (command === "/help") {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Ngài chọn chức năng, sau đó gửi văn bản, PDF tối đa 2 trang hoặc một ảnh. Dùng /cancel để hủy và /start để trở về menu chính.",
    });
    return;
  }
  if (command) {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Trợ lý Dochat chưa hiểu lệnh này. Ngài dùng /start để mở menu chính.",
    });
    return;
  }

  const stateValue = await env.CONVERSATIONS.get(`state:${userId}`);
  if (!stateValue) {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Ngài vui lòng chọn chức năng trước.",
      reply_markup: homeKeyboard,
    });
    return;
  }
  const state = JSON.parse(stateValue) as Conversation;
  const source = sourceFromMessage(message);
  if (!source) {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Trợ lý Dochat chỉ hỗ trợ văn bản, PDF tối đa 2 trang hoặc ảnh JPG, PNG, WEBP.",
    });
    return;
  }
  if ((source.file_size || 0) > MAX_FILE_BYTES) {
    await telegram(env, "sendMessage", {
      chat_id: chatId,
      text: "Tệp vượt quá giới hạn 15 MB. Ngài vui lòng gửi tệp nhỏ hơn.",
    });
    return;
  }

  const jobKey = `job:${update.update_id}`;
  if (await env.CONVERSATIONS.get(jobKey)) return;
  const status = await telegram<{ message_id: number }>(env, "sendMessage", {
    chat_id: chatId,
    text: "Trợ lý Dochat đã nhận nội dung và đang xử lý...",
  });
  const job: Job = {
    update_id: update.update_id,
    chat_id: chatId,
    user_id: userId,
    source_type: source.source_type,
    mode: state.mode,
    file_id: source.file_id,
    text: source.text,
    status_message_id: status.message_id,
  };
  await env.CONVERSATIONS.put(jobKey, "queued", { expirationTtl: JOB_TTL_SECONDS });
  try {
    await env.JOBS.send(job);
  } catch (error) {
    await env.CONVERSATIONS.delete(jobKey);
    throw error;
  }
  await env.CONVERSATIONS.delete(`state:${userId}`);
}

async function handleUpdate(update: any, env: Env): Promise<void> {
  if (update.callback_query) return handleCallback(update, env);
  if (update.message) return handleMessage(update, env);
}

async function updateStatus(env: Env, job: Job, text: string): Promise<void> {
  if (!job.status_message_id) return;
  try {
    await telegram(env, "editMessageText", {
      chat_id: job.chat_id,
      message_id: job.status_message_id,
      text,
    });
  } catch {
    // The final result remains usable if a progress message cannot be edited.
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/internal/normalize" && request.method === "POST") {
      if (request.headers.get("X-Internal-Secret") !== env.INTERNAL_SECRET) {
        return new Response("Unauthorized", { status: 401 });
      }
      const { text } = (await request.json()) as { text?: string };
      if (!text) return new Response("Invalid request", { status: 400 });
      const result = await env.AI.run(
        env.AI_MODEL || "@cf/meta/llama-3.1-8b-instruct-fast",
        {
          messages: [{ role: "user", content: normalizePrompt(text) }],
          max_new_tokens: 4096,
        },
      );
      return Response.json({ text: (result as { response?: string }).response || "" });
    }

    if (url.pathname === "/internal/summarize" && request.method === "POST") {
      if (request.headers.get("X-Internal-Secret") !== env.INTERNAL_SECRET) {
        return new Response("Unauthorized", { status: 401 });
      }
       const { prompt, max_tokens } = (await request.json()) as {
        prompt?: string;
        max_tokens?: number;
      };
      if (!prompt) return new Response("Invalid request", { status: 400 });
      const result = await env.AI.run(
        env.AI_MODEL || "@cf/meta/llama-3.1-8b-instruct-fast",
        {
          messages: [{ role: "user", content: prompt }],
          max_new_tokens: max_tokens ?? 1024,
        },
      );
      return Response.json({ text: (result as { response?: string }).response || "" });
    }

    if (url.pathname !== "/webhook" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.TELEGRAM_WEBHOOK_SECRET) {
      return new Response("Unauthorized", { status: 401 });
    }

    const update = (await request.json()) as { update_id?: number };
    if (update.update_id === undefined) return Response.json({ ok: true });
    const updateKey = `update:${update.update_id}`;
    if (await env.CONVERSATIONS.get(updateKey)) return Response.json({ ok: true });
    try {
      await handleUpdate(update, env);
    } catch (error) {
      console.error("webhook_handle_failed", error);
    }
    await env.CONVERSATIONS.put(updateKey, "handled", { expirationTtl: JOB_TTL_SECONDS });
    return Response.json({ ok: true });
  },

  async queue(batch: MessageBatch<Job>, env: Env): Promise<void> {
    for (const message of batch.messages) {
      const job = message.body;
      const key = `job:${job.update_id}`;
      if ((await env.CONVERSATIONS.get(key)) === "done") {
        message.ack();
        continue;
      }
      try {
        const response = await fetch(`${env.BACKEND_URL.replace(/\/$/, "")}/process`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "X-Internal-Secret": env.INTERNAL_SECRET,
          },
          body: JSON.stringify(job),
        });
        if (response.ok) {
          const result = (await response.json()) as {
            status: "completed" | "rejected" | "awaiting_approval";
            user_message?: string;
            error_id?: string;
          };
          if (result.status === "rejected") {
            await updateStatus(
              env,
              job,
              `${result.user_message || "Không thể xử lý nội dung này."}\nMã lỗi: ${result.error_id || "UNKNOWN"}`,
            );
          }
          await env.CONVERSATIONS.put(key, "done", { expirationTtl: JOB_TTL_SECONDS });
          message.ack();
          continue;
        }
      } catch {
        // Retry transient Render/network failures below.
      }

      if (message.attempts >= 3) {
        await updateStatus(
          env,
          job,
          "Trợ lý Dochat chưa thể xử lý nội dung này. Ngài vui lòng thử lại sau.",
        );
        await env.CONVERSATIONS.put(key, "failed", { expirationTtl: JOB_TTL_SECONDS });
        message.ack();
      } else {
        message.retry({ delaySeconds: 30 * message.attempts });
      }
    }
  },
};
