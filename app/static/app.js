/* RAG Agent 演示台：同源调用 /api/v1，无框架、无构建。 */
'use strict';

const API = '/api/v1';
const LS = { token: 'rag.token', user: 'rag.user', kb: 'rag.kbId' };

const state = {
  token: localStorage.getItem(LS.token) || '',
  user: JSON.parse(localStorage.getItem(LS.user) || 'null'),
  kbs: [],
  kbId: Number(localStorage.getItem(LS.kb)) || null,
  docs: [],
  convId: null,
  messages: [],
  streaming: false,
  pollers: new Set(),
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

function toast(message, kind = 'info') {
  const node = document.createElement('div');
  node.className = `toast toast-${kind}`;
  node.textContent = message;
  $('toasts').appendChild(node);
  setTimeout(() => node.remove(), kind === 'err' ? 6000 : 3600);
}

function saveAuth(token, user) {
  state.token = token || '';
  state.user = user || null;
  if (token) {
    localStorage.setItem(LS.token, token);
    localStorage.setItem(LS.user, JSON.stringify(user));
  } else {
    localStorage.removeItem(LS.token);
    localStorage.removeItem(LS.user);
  }
}

/** 统一请求：带 token、解析 detail 错误、401 自动登出。 */
async function api(path, { method = 'GET', json, form, body, raw = false } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  let payload = body;
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(json);
  } else if (form !== undefined) {
    headers['Content-Type'] = 'application/x-www-form-urlencoded';
    payload = new URLSearchParams(form).toString();
  }

  const resp = await fetch(`${API}${path}`, { method, headers, body: payload });
  if (resp.status === 401) {
    saveAuth('', null);
    showAuth();
    throw new Error('登录已过期，请重新登录');
  }
  if (!resp.ok) throw new Error(await errorText(resp));
  if (raw) return resp;
  return resp.status === 204 ? null : resp.json();
}

async function errorText(resp) {
  let detail = `请求失败（HTTP ${resp.status}）`;
  try {
    const data = await resp.json();
    if (typeof data.detail === 'string') detail = data.detail;
    else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg || '').join('；');
  } catch { /* 忽略非 JSON 响应体 */ }
  if (resp.status === 429) detail = `触发限流：${detail}`;
  return detail;
}

/* ---------- 视图切换 ---------- */

function showAuth() {
  state.convId = null;
  state.messages = [];
  state.docs = [];
  $('auth-view').classList.remove('hidden');
  $('main-view').classList.add('hidden');
  $('user-chip').classList.add('hidden');
  $('logout-btn').classList.add('hidden');
}

function showMain() {
  $('auth-view').classList.add('hidden');
  $('main-view').classList.remove('hidden');
  $('user-chip').classList.remove('hidden');
  $('logout-btn').classList.remove('hidden');
  $('user-chip').textContent = state.user ? `${state.user.name} · ${state.user.email}` : '';
}

/* ---------- 认证 ---------- */

async function doLogin(email, password) {
  const data = await api('/auth/login', { method: 'POST', form: { username: email, password } });
  saveAuth(data.access_token, data.user);
  showMain();
  await afterLogin();
  toast(`欢迎回来，${data.user.name}`, 'ok');
}

async function doRegister(email, name, password) {
  await api('/auth/register', { method: 'POST', json: { email, name, password } });
  await doLogin(email, password);
}

async function afterLogin() {
  await loadKbs();
  loadSuggestions();
  healthTick();
}

/* ---------- 知识库 ---------- */

async function loadKbs() {
  state.kbs = await api('/knowledge-bases');
  if (state.kbId && !state.kbs.some((kb) => kb.id === state.kbId)) state.kbId = null;
  if (!state.kbId && state.kbs.length) state.kbId = state.kbs[0].id;
  localStorage.setItem(LS.kb, state.kbId || '');
  renderKbs();
  if (state.kbId) await loadDocs();
  else renderDocs();
}

function renderKbs() {
  const list = $('kb-list');
  list.innerHTML = '';
  $('kb-empty').classList.toggle('hidden', state.kbs.length > 0);

  for (const kb of state.kbs) {
    const li = document.createElement('li');
    li.className = `item${kb.id === state.kbId ? ' active' : ''}`;
    li.innerHTML = `
      <div class="item-row">
        <span class="item-name" title="${esc(kb.name)}">${esc(kb.name)}</span>
        <button class="btn btn-icon" data-del="${kb.id}" title="删除知识库">×</button>
      </div>
      <div class="item-meta">
        <span>${kb.document_count} 文档</span>
        <span class="status status-ready">${kb.ready_count} 就绪</span>
      </div>`;
    li.addEventListener('click', (ev) => {
      if (ev.target.dataset.del) return;
      if (kb.id !== state.kbId) {
        state.kbId = kb.id;
        state.convId = null;
        localStorage.setItem(LS.kb, kb.id);
        renderKbs();
        loadDocs();
      }
    });
    li.querySelector('[data-del]').addEventListener('click', (ev) => {
      ev.stopPropagation();
      confirmModal('删除知识库', `「${kb.name}」及其全部文档、分块都会被删除，且不可恢复。`, async () => {
        try {
          await api(`/knowledge-bases/${kb.id}`, { method: 'DELETE' });
          if (state.kbId === kb.id) state.kbId = null;
          await loadKbs();
          renderDocs();
          toast('知识库已删除', 'ok');
        } catch (err) {
          toast(err.message, 'err');
        }
      });
    });
    list.appendChild(li);
  }
}

/* ---------- 文档 ---------- */

async function loadDocs() {
  state.docs = await api(`/knowledge-bases/${state.kbId}/documents`);
  renderDocs();
  state.docs.filter((d) => d.status === 'pending' || d.status === 'processing').forEach((d) => pollDoc(d.id));
}

function renderDocs() {
  const list = $('doc-list');
  list.innerHTML = '';
  const kb = state.kbs.find((k) => k.id === state.kbId);
  $('docs-title').textContent = kb ? `文档 · ${kb.name}` : '文档';
  $('doc-empty').classList.toggle('hidden', state.docs.length > 0);
  $('doc-empty').textContent = state.kbId ? '还没有文档，点「+ 上传」添加' : '选择知识库后可上传文档';

  for (const doc of state.docs) {
    const li = document.createElement('li');
    li.className = 'item';
    li.innerHTML = `
      <div class="item-row">
        <span class="item-name" title="${esc(doc.filename)}">${esc(doc.filename)}</span>
        <span class="item-actions">
          <button class="btn btn-icon" data-view title="查看分块内容">查看</button>
          <button class="btn btn-icon" data-del title="删除文档">×</button>
        </span>
      </div>
      <div class="item-meta">
        <span class="status status-${esc(doc.status)}">${esc(doc.status)}</span>
        <span>${doc.chunk_count} chunks</span>
        <span>${(doc.file_size / 1024).toFixed(1)} KB</span>
      </div>
      ${doc.error_message ? `<div class="item-meta" style="color:var(--err)">${esc(doc.error_message)}</div>` : ''}`;
    li.querySelector('[data-view]').addEventListener('click', () => openDocChunks(doc));
    li.querySelector('[data-del]').addEventListener('click', () => {
      confirmModal('删除文档', `删除「${doc.filename}」及其全部分块？该文档将不再参与检索。`, async () => {
        try {
          await api(`/documents/${doc.id}`, { method: 'DELETE' });
          await loadDocs();
          toast('文档已删除', 'ok');
        } catch (err) {
          toast(err.message, 'err');
        }
      });
    });
    list.appendChild(li);
  }
}

/** 轮询摄取状态，直到 ready / failed（Celery eager 模式下上传时已就绪，直接跳过）。 */
function pollDoc(docId) {
  if (state.pollers.has(docId)) return;
  state.pollers.add(docId);
  let tries = 0;
  const tick = async () => {
    tries += 1;
    try {
      const st = await api(`/documents/${docId}/status`);
      const doc = state.docs.find((d) => d.id === docId);
      if (doc) {
        doc.status = st.status;
        doc.chunk_count = st.chunk_count;
        doc.error_message = st.error_message;
        renderDocs();
      }
      if (st.status === 'ready') {
        state.pollers.delete(docId);
        toast(`《${doc ? doc.filename : docId}》已就绪：${st.chunk_count} chunks`, 'ok');
        await loadKbs();
        return;
      }
      if (st.status === 'failed') {
        state.pollers.delete(docId);
        toast(`摄入失败：${st.error_message || '未知原因'}`, 'err');
        return;
      }
    } catch (err) {
      state.pollers.delete(docId);
      toast(err.message, 'err');
      return;
    }
    if (tries >= 40) {
      state.pollers.delete(docId);
      toast('文档仍在处理中，请稍后刷新', 'err');
      return;
    }
    setTimeout(tick, 1500);
  };
  setTimeout(tick, 1200);
}

async function uploadFiles(files) {
  if (!state.kbId) return toast('请先选择知识库', 'err');
  for (const file of files) {
    const form = new FormData();
    form.append('file', file);
    try {
      const doc = await api(`/knowledge-bases/${state.kbId}/documents`, { method: 'POST', body: form });
      state.docs.unshift(doc);
      renderDocs();
      if (doc.status === 'ready') toast(`《${doc.filename}》已就绪：${doc.chunk_count} chunks`, 'ok');
      else pollDoc(doc.id);
    } catch (err) {
      toast(`《${file.name}》上传失败：${err.message}`, 'err');
    }
  }
  await loadKbs();
}

/* ---------- 弹窗 ---------- */

function openModal({ title, body, actions = [] }) {
  $('modal-title').textContent = title;
  const bodyBox = $('modal-body');
  bodyBox.replaceChildren(...(body ? [].concat(body) : []));
  const actionsBox = $('modal-actions');
  actionsBox.replaceChildren();
  for (const action of actions) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn${action.primary ? ' btn-primary' : ''}`;
    btn.textContent = action.label;
    btn.addEventListener('click', () => action.onClick());
    actionsBox.appendChild(btn);
  }
  $('modal').classList.remove('hidden');
  bodyBox.scrollTop = 0;
}

function closeModal() {
  $('modal').classList.add('hidden');
  $('modal-body').replaceChildren();
  $('modal-actions').replaceChildren();
}

function confirmModal(title, message, onConfirm) {
  const p = document.createElement('p');
  p.className = 'hint';
  p.textContent = message;
  openModal({
    title,
    body: p,
    actions: [
      { label: '取消', onClick: closeModal },
      { label: '确认删除', primary: true, onClick: async () => { closeModal(); await onConfirm(); } },
    ],
  });
}

function kbCreateModal() {
  const wrap = document.createElement('div');
  wrap.className = 'form';
  wrap.innerHTML = `
    <label>知识库名称
      <input id="kb-name-input" maxlength="128" placeholder="例如：员工手册库">
    </label>
    <label>描述（可选）
      <textarea id="kb-desc-input" rows="3" maxlength="2000" placeholder="演示用知识库"></textarea>
    </label>`;

  const submit = async () => {
    const name = wrap.querySelector('#kb-name-input').value.trim();
    if (!name) return toast('请填写知识库名称', 'err');
    try {
      const kb = await api('/knowledge-bases', {
        method: 'POST',
        json: { name, description: wrap.querySelector('#kb-desc-input').value.trim() || null },
      });
      state.kbId = kb.id;
      state.convId = null;
      closeModal();
      await loadKbs();
      toast(`知识库「${kb.name}」已创建，可以上传文档了`, 'ok');
    } catch (err) {
      toast(err.message, 'err');
    }
  };

  wrap.querySelector('#kb-name-input').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); submit(); }
  });
  openModal({
    title: '新建知识库',
    body: wrap,
    actions: [{ label: '取消', onClick: closeModal }, { label: '创建', primary: true, onClick: submit }],
  });
  wrap.querySelector('#kb-name-input').focus();
}

/** 查看文档解析出的分块内容（回答引用的就是这些片段）。 */
async function openDocChunks(doc) {
  const loading = document.createElement('p');
  loading.className = 'hint';
  loading.textContent = '加载中…';
  openModal({
    title: `${doc.filename} · 分块内容`,
    body: loading,
    actions: [{ label: '关闭', onClick: closeModal }],
  });

  const frag = document.createDocumentFragment();
  const meta = document.createElement('p');
  meta.className = 'hint';
  meta.textContent = `${doc.filename} · ${doc.status} · ${doc.chunk_count} 块 · ${(doc.file_size / 1024).toFixed(1)} KB`;
  frag.appendChild(meta);

  try {
    const chunks = await api(`/documents/${doc.id}/chunks`);
    if (!chunks.length) {
      const empty = document.createElement('p');
      empty.className = 'hint';
      empty.textContent = doc.status === 'failed'
        ? `摄入失败：${doc.error_message || '未知原因'}`
        : '还没有分块：文档可能仍在处理中，或分块结果为空。';
      frag.appendChild(empty);
    }
    for (const chunk of chunks) {
      const node = document.createElement('div');
      node.className = 'chunk';
      const head = document.createElement('div');
      head.className = 'chunk-head';
      const seq = document.createElement('span');
      seq.textContent = `#${chunk.seq + 1}`;
      const size = document.createElement('span');
      size.textContent = `${chunk.content.length} 字`;
      head.append(seq, size);
      const pre = document.createElement('pre');
      pre.textContent = chunk.content;
      node.append(head, pre);
      frag.appendChild(node);
    }
    $('modal-body').replaceChildren(frag);
  } catch (err) {
    loading.textContent = `加载失败：${err.message}`;
  }
}

/* ---------- 问答（SSE） ---------- */

async function ensureConversation() {
  if (state.convId) return state.convId;
  const conv = await api('/conversations', { method: 'POST', json: { knowledge_base_id: state.kbId } });
  state.convId = conv.id;
  $('conv-chip').textContent = `会话 #${conv.id}`;
  $('conv-chip').classList.remove('hidden');
  return conv.id;
}

function pushMessage(message) {
  state.messages.push(message);
  renderMessages();
  return message;
}

function renderMessages() {
  const box = $('messages');
  box.innerHTML = '';
  if (!state.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '选中知识库，就库里的内容提问；回答会带 [1][2] 引用标注。';
    box.appendChild(empty);
    return;
  }

  state.messages.forEach((msg, index) => {
    const wrap = document.createElement('div');
    wrap.className = `msg msg-${msg.role === 'user' ? 'user' : 'bot'}${msg.error ? ' msg-err' : ''}`;
    wrap.dataset.index = index;
    wrap.innerHTML = `<div class="bubble"><span class="text"></span></div>`;
    wrap.querySelector('.text').innerHTML = msgHtml(msg, index);
    if (msg.sources && msg.sources.length) wrap.querySelector('.bubble').appendChild(sourcesNode(msg.sources, index));
    box.appendChild(wrap);
  });
  box.scrollTop = box.scrollHeight;
}

function msgHtml(msg, index) {
  const text = esc(msg.content);
  const withCites = text.replace(/\[(\d{1,2})\]/g, (m, n) => `<a class="cite-link" data-cite="${n}" data-owner="${index}">[${n}]</a>`);
  const cursor = msg.pending ? '<span class="cursor"></span>' : '';
  const badges = [];
  if (msg.cached) badges.push('<span class="badge badge-cache">缓存命中</span>');
  if (msg.sources && msg.sources.length) badges.push(`<span class="badge">${msg.sources.length} 条引用</span>`);
  if (msg.error) badges.push('<span class="badge">生成中断</span>');
  const badgeHtml = badges.length ? `<div class="msg-badges">${badges.join('')}</div>` : '';
  return `${withCites}${cursor}${badgeHtml}`;
}

function sourcesNode(sources, owner) {
  const box = document.createElement('div');
  box.className = 'sources';
  sources.forEach((src, i) => {
    const node = document.createElement('div');
    node.className = 'source';
    node.dataset.cite = String(i + 1);
    node.dataset.owner = String(owner);
    node.innerHTML = `
      <div class="source-head">
        <span class="cite-link">[${i + 1}]</span>
        <span class="source-name" title="${esc(src.filename)}">${esc(src.filename || '未知来源')}</span>
        <span class="source-score">score ${Number(src.score ?? 0).toFixed(4)}</span>
      </div>
      <p class="source-snippet">${esc(src.snippet || '')}</p>`;
    box.appendChild(node);
  });
  return box;
}

/** 点 [n] 角标 → 高亮对应引用卡片。 */
function bindCitationJump() {
  $('messages').addEventListener('click', (ev) => {
    const link = ev.target.closest('.cite-link[data-cite]');
    if (!link) return;
    const owner = link.dataset.owner;
    const card = $('messages').querySelector(`.source[data-owner="${owner}"][data-cite="${link.dataset.cite}"]`);
    if (!card) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1200);
  });
}

/** EventSource 不支持 POST，这里用 fetch 读 SSE 流。 */
function streamAnswer(convId, question) {
  return fetch(`${API}/conversations/${convId}/ask-stream`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${state.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
}

async function ask(question) {
  if (!state.kbId) return toast('请先选择知识库', 'err');
  if (state.streaming) return;

  state.streaming = true;
  $('send-btn').disabled = true;
  $('question').value = '';
  pushMessage({ role: 'user', content: question });

  const answer = pushMessage({ role: 'assistant', content: '', sources: [], pending: true });
  const answerIndex = state.messages.length - 1;
  const bubble = $('messages').querySelector(`.msg[data-index="${answerIndex}"] .text`);

  try {
    let convId = await ensureConversation();
    let resp = await streamAnswer(convId, question);
    if (resp.status === 404) {
      // 会话已被删除（清库 / 误删）：重建会话再试一次
      state.convId = null;
      convId = await ensureConversation();
      resp = await streamAnswer(convId, question);
    }
    if (resp.status === 401) { saveAuth('', null); showAuth(); throw new Error('登录已过期，请重新登录'); }
    if (!resp.ok) throw new Error(await errorText(resp));
    if (!resp.body) throw new Error('当前浏览器不支持流式读取');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let sourcesDirty = false;

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';
      for (const chunk of events) {
        const line = chunk.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        if (event.type === 'sources') {
          answer.sources = event.sources || [];
          sourcesDirty = true;
        } else if (event.type === 'delta') {
          answer.content += event.text || '';
          bubble.innerHTML = msgHtml(answer, answerIndex);
        } else if (event.type === 'done') {
          answer.pending = false;
          answer.cached = !!event.cached;
        } else if (event.type === 'error') {
          answer.pending = false;
          answer.error = true;
          answer.content += answer.content ? '\n\n[生成中断] ' : '[生成中断] ';
          toast(event.detail || '生成回答时出错', 'err');
        }
      }
    }
    answer.pending = false;
    renderMessages();
    if (sourcesDirty) toast(`检索到 ${answer.sources.length} 条引用片段`, 'ok');
  } catch (err) {
    answer.pending = false;
    answer.error = true;
    if (!answer.content) answer.content = `请求失败：${err.message}`;
    renderMessages();
    toast(err.message, 'err');
  } finally {
    state.streaming = false;
    $('send-btn').disabled = false;
    $('question').focus();
  }
}

/* ---------- 健康检查与示例问题 ---------- */

async function healthTick() {
  const pill = $('health-pill');
  try {
    const data = await api('/health');
    const ok = data.status === 'ok';
    pill.className = `pill ${ok ? 'pill-ok' : 'pill-warn'}`;
    pill.textContent = `后端 ${data.status} · db:${data.checks.db} redis:${data.checks.redis}`;
  } catch (err) {
    pill.className = 'pill pill-err';
    pill.textContent = '后端不可用';
  }
}

async function loadSuggestions() {
  try {
    const resp = await fetch('/demo/sample-questions.json');
    const questions = await resp.json();
    const box = $('suggestions');
    box.innerHTML = '';
    questions.slice(0, 8).forEach((q) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip-btn';
      btn.textContent = q;
      btn.addEventListener('click', () => { $('question').value = q; $('question').focus(); });
      box.appendChild(btn);
    });
  } catch { /* 示例问题缺失不影响主流程 */ }
}

/* ---------- 事件绑定与启动 ---------- */

function bind() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      $('login-form').classList.toggle('hidden', tab.dataset.tab !== 'login');
      $('register-form').classList.toggle('hidden', tab.dataset.tab !== 'register');
    });
  });

  $('login-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const data = new FormData(ev.target);
    try {
      await doLogin(data.get('email').trim(), data.get('password'));
    } catch (err) {
      toast(err.message, 'err');
    }
  });

  $('register-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const data = new FormData(ev.target);
    try {
      await doRegister(data.get('email').trim(), data.get('name').trim(), data.get('password'));
    } catch (err) {
      toast(err.message, 'err');
    }
  });

  $('logout-btn').addEventListener('click', () => {
    saveAuth('', null);
    showAuth();
    $('health-pill').className = 'pill pill-muted';
    toast('已退出登录');
  });

  $('new-kb-btn').addEventListener('click', kbCreateModal);
  $('modal-close').addEventListener('click', closeModal);
  $('modal').addEventListener('click', (ev) => {
    if (ev.target === $('modal')) closeModal();
  });

  $('file-input').addEventListener('change', async (ev) => {
    const files = Array.from(ev.target.files || []);
    ev.target.value = '';
    if (files.length) await uploadFiles(files);
  });

  $('new-conv-btn').addEventListener('click', () => {
    state.convId = null;
    state.messages = [];
    $('conv-chip').classList.add('hidden');
    renderMessages();
    toast('已开启新对话');
  });

  $('ask-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const q = $('question').value.trim();
    if (!q) return;
    ask(q);
  });

  $('question').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      $('ask-form').requestSubmit();
    }
  });

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && !$('modal').classList.contains('hidden')) closeModal();
  });

  bindCitationJump();
}

async function main() {
  bind();
  renderMessages();
  setInterval(healthTick, 30000);

  if (!state.token) {
    showAuth();
    healthTick();
    return;
  }
  showMain();
  try {
    await afterLogin();
  } catch (err) {
    toast(err.message, 'err');
    showAuth();
  }
}

main();
