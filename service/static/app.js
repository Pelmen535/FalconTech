'use strict';
/* Тонкий клиент: только транспорт и отрисовка. Ни одного решения о сходстве здесь нет —
   порог, ранжирование и отказ приходят с сервера и показываются как есть. */

const $ = id => document.getElementById(id);
const TABS = ['search', 'gallery', 'about'];
const PAGE = 24;

let apiKey = '';
let busy = false;
let offset = 0;
let galleryTotal = 0;
let lastResult = null;          // для выгрузки CSV/JSON
let lastQuery = null;           // файл и рамка последнего запроса — нужны объяснению

const objectUrls = new Map();
const pendingImages = new Map();
let imageGeneration = 0;

// ----------------------------------------------------------------- инфраструктура
function message(text, error = false) {
  const node = $('message');
  node.textContent = text;
  node.classList.toggle('error', error);
  node.classList.toggle('hidden', !text);
}

function describeError(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(x => x.msg || JSON.stringify(x)).join('; ');
  return JSON.stringify(value);
}

async function api(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (apiKey) headers.set('X-API-Key', apiKey);
  let response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch {
    throw new Error('Нет связи с сервером. Проверьте, что сервис запущен.');
  }
  if (!response.ok) {
    let detail;
    try { detail = describeError((await response.json()).detail); }
    catch { detail = 'Ошибка сервера: ' + response.status; }
    throw new Error(detail || 'Ошибка HTTP ' + response.status);
  }
  return response;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* Картинки запрашиваются через fetch, а не через <img src>: иначе браузер не пришлёт
   заголовок X-API-Key и защищённый экземпляр показывал бы пустые плитки. */
function resetImageCache() {
  imageGeneration++;
  for (const url of objectUrls.values()) URL.revokeObjectURL(url);
  objectUrls.clear();
  pendingImages.clear();
}

async function attachImage(img, url) {
  const generation = imageGeneration;
  try {
    let blobUrl = objectUrls.get(url);
    if (!blobUrl) {
      let request = pendingImages.get(url);
      if (!request) {
        request = api(url).then(r => r.blob()).then(blob => {
          if (generation !== imageGeneration) return null;
          const value = URL.createObjectURL(blob);
          objectUrls.set(url, value);
          return value;
        }).finally(() => { if (generation === imageGeneration) pendingImages.delete(url); });
        pendingImages.set(url, request);
      }
      blobUrl = await request;
    }
    if (blobUrl && generation === imageGeneration && img.isConnected) img.src = blobUrl;
  } catch { img.alt = 'Изображение недоступно'; }
}

async function run(action) {
  if (busy) return;
  busy = true;
  document.querySelectorAll('button, input').forEach(node => { node.disabled = true; });
  try { await action(); }
  catch (error) { message(error.message, true); }
  finally {
    busy = false;
    document.querySelectorAll('button, input').forEach(node => { node.disabled = false; });
    pagination();
  }
}

function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ----------------------------------------------------------------- выбор рамки
/* Рамку можно и ввести числами, и протянуть мышью прямо по кадру. Числа и холст —
   один источник истины: поля хранят пиксели исходного кадра, холст только рисует. */
function boxPicker(fileId, canvasId, wrapId, boxId) {
  const file = $(fileId), canvas = $(canvasId), wrap = $(wrapId), box = $(boxId);
  const inputs = Array.from(box.querySelectorAll('input'));
  let image = null, generation = 0, ratio = 1, dragging = null;

  const values = () => inputs.map(node => Number(node.value));
  const setValues = list => inputs.forEach((node, i) => { node.value = Math.round(list[i]); });

  function draw() {
    if (!image) return;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const [x, y, w, h] = values();
    if (![x, y, w, h].every(Number.isFinite)) return;
    context.save();
    context.strokeStyle = '#8ce5c0';
    context.lineWidth = 3;
    context.setLineDash([]);
    context.strokeRect(x * ratio, y * ratio, w * ratio, h * ratio);
    context.fillStyle = 'rgba(140,229,192,.12)';
    context.fillRect(x * ratio, y * ratio, w * ratio, h * ratio);
    context.restore();
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;                 // холст масштабируется по CSS
    return {
      x: Math.min(Math.max((event.clientX - rect.left) * scale / ratio, 0), image.naturalWidth),
      y: Math.min(Math.max((event.clientY - rect.top) * scale / ratio, 0), image.naturalHeight),
    };
  }

  canvas.addEventListener('pointerdown', event => {
    if (!image || busy) return;
    canvas.setPointerCapture(event.pointerId);
    dragging = point(event);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const now = point(event);
    setValues([Math.min(dragging.x, now.x), Math.min(dragging.y, now.y),
               Math.abs(now.x - dragging.x), Math.abs(now.y - dragging.y)]);
    draw();
  });
  const finish = () => {
    if (!dragging) return;
    dragging = null;
    const [, , w, h] = values();
    if (w < 8 || h < 8) { setValues([0, 0, image.naturalWidth, image.naturalHeight]); draw(); }
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);

  file.addEventListener('change', async () => {
    const current = ++generation;
    image = null;
    wrap.classList.add('hidden');
    if (!file.files[0]) return;
    const url = URL.createObjectURL(file.files[0]);
    const picture = new Image();
    try {
      picture.src = url;
      await picture.decode();
      if (current !== generation) return;
      image = picture;
      ratio = Math.min(1, 1000 / picture.naturalWidth);
      canvas.width = Math.round(picture.naturalWidth * ratio);
      canvas.height = Math.round(picture.naturalHeight * ratio);
      setValues([0, 0, picture.naturalWidth, picture.naturalHeight]);
      wrap.classList.remove('hidden');
      draw();
    } catch {
      message('Не удалось открыть изображение. Используйте JPEG, PNG или WebP.', true);
    } finally { URL.revokeObjectURL(url); }
  });
  box.addEventListener('input', draw);

  function read() {
    if (!image || !file.files[0]) throw new Error('Выберите изображение и дождитесь предпросмотра.');
    const b = values();
    if (!b.every(Number.isFinite) || b[0] < 0 || b[1] < 0 || b[2] <= 0 || b[3] <= 0 ||
        b[0] + b[2] > image.naturalWidth + 1e-6 || b[1] + b[3] > image.naturalHeight + 1e-6) {
      throw new Error('Рамка должна лежать внутри кадра и иметь положительный размер.');
    }
    return b;
  }

  /* Подставить готовый кадр программно (демонстрационный запрос). Идём тем же путём,
     что и ручной выбор файла: DataTransfer + событие change, чтобы не было второй
     ветки логики, которую забудут починить. */
  async function load(blob, name, bbox) {
    const transfer = new DataTransfer();
    transfer.items.add(new File([blob], name, { type: blob.type || 'image/jpeg' }));
    file.files = transfer.files;
    file.dispatchEvent(new Event('change'));
    for (let attempt = 0; attempt < 100 && !image; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    if (image && Array.isArray(bbox) && bbox.length === 4) { setValues(bbox); draw(); }
  }

  return { read, load };
}

// ----------------------------------------------------------------- состояние
async function health() {
  const state = await (await api('/health')).json();
  galleryTotal = state.gallery_count;
  $('gallery-count').textContent = galleryTotal;
  $('health').textContent = (state.busy ? 'Обработка запроса' : 'Сервис доступен') +
    ' · ' + galleryTotal + ' ' + plural(galleryTotal, 'наблюдение', 'наблюдения', 'наблюдений');
  return state;
}

function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b === 1) return one;
  return b >= 2 && b <= 4 ? few : many;
}

function pagination() {
  $('prev').disabled = busy || offset === 0;
  $('next').disabled = busy || offset + PAGE >= galleryTotal;
  $('page-info').textContent = galleryTotal
    ? (offset + 1) + '–' + Math.min(offset + PAGE, galleryTotal) + ' из ' + galleryTotal
    : '';
}

// ----------------------------------------------------------------- карточки
function card(item, target, options = {}) {
  const box = element('article', 'card' + (options.accepted && item.rank === 1 ? ' top' : ''));
  const img = element('img');
  img.alt = 'Автомобиль ' + item.image_id;
  img.loading = 'lazy';
  box.append(img);

  const body = element('div', 'card-body');
  if (item.rank !== undefined) {
    body.append(element('strong', '', '№ ' + item.rank + ' · ' + Number(item.cosine).toFixed(3)));
    const bar = element('div', 'bar');
    const fill = element('i');
    // Косинус в [-1, 1]; показываем положительную часть — отрицательных у похожих не бывает.
    fill.style.width = Math.max(0, Math.min(1, Number(item.cosine))) * 100 + '%';
    bar.append(fill);
    body.append(bar);
  }
  const id = element('div', 'card-id', item.image_id);
  id.title = item.image_id;
  body.append(id);

  if (options.explainable) {
    const button = element('button', '', 'Почему похоже');
    button.type = 'button';
    button.addEventListener('click', () => run(() => explain(item)));
    body.append(button);
  }
  if (options.deletable) {
    const button = element('button', 'danger', 'Удалить');
    button.type = 'button';
    button.addEventListener('click', () => {
      if (!confirm('Удалить наблюдение «' + item.image_id + '» из галереи?')) return;
      run(async () => {
        await api('/v1/gallery/items/' + encodeURIComponent(item.image_id), { method: 'DELETE' });
        clearResults(); resetImageCache();
        await loadGallery(); await health();
        message('Наблюдение удалено.');
      });
    });
    body.append(button);
  }
  box.append(body);
  target.append(box);
  attachImage(img, item.thumbnail_url || item.image_url);
}

function clearResults() {
  $('results').replaceChildren();
  $('result-info').className = 'hidden';
  $('result-empty').classList.remove('hidden');
  $('export-buttons').classList.add('hidden');
  lastResult = null;
}

// ----------------------------------------------------------------- галерея
async function loadGallery() {
  const page = await (await api('/v1/gallery?offset=' + offset + '&limit=' + PAGE)).json();
  galleryTotal = page.count;
  if (!page.items.length && offset > 0 && galleryTotal <= offset) {
    offset = Math.max(0, Math.floor((galleryTotal - 1) / PAGE) * PAGE);
    return loadGallery();
  }
  $('gallery-count').textContent = galleryTotal;
  $('gallery').replaceChildren();
  $('gallery-empty').classList.toggle('hidden', page.items.length !== 0);
  for (const item of page.items) card(item, $('gallery'), { deletable: true });
  pagination();
}

// ----------------------------------------------------------------- объяснение
async function explain(item) {
  if (!lastQuery) throw new Error('Сначала выполните поиск: объяснение считается для его запроса.');
  const dialog = $('explain-dialog');
  $('explain-title').textContent = 'Почему похоже';
  $('explain-content').replaceChildren(element('p', 'muted', 'Строим карту сходства…'));
  if (!dialog.open) dialog.showModal();

  const body = new FormData();
  body.append('image', lastQuery.file);
  body.append('bbox', JSON.stringify(lastQuery.bbox));
  body.append('gallery_id', item.image_id);
  const answer = await (await api('/v1/explain', { method: 'POST', body })).json();

  const content = $('explain-content');
  content.replaceChildren();
  const titles = { query: 'Запрос', candidate: 'Кандидат № ' + item.rank };
  for (const name of ['query', 'candidate']) {
    const side = answer.sides[name];
    const figure = element('figure');
    const img = element('img');
    img.src = 'data:image/png;base64,' + side.heatmap_png_base64;
    img.alt = 'Карта сходства: ' + titles[name];
    const caption = element('figcaption');
    caption.textContent = titles[name];
    figure.append(img, caption);
    content.append(figure);
  }
  const note = element('p', 'summary');
  note.textContent = 'Уверенность ' + answer.cosine.toFixed(3) + ' при пороге ' +
    answer.threshold.toFixed(2) + '.';
  content.append(note);
}

// ----------------------------------------------------------------- выгрузка
function exportRows() {
  const rows = [['query_id', 'rank', 'gallery_id', 'cosine', 'accepted', 'threshold', 'search_mode']];
  const queryId = lastQuery && lastQuery.id ? lastQuery.id : '(загруженный кадр)';
  for (const item of lastResult.ranking) {
    rows.push([queryId, item.rank, item.image_id, item.cosine.toFixed(6),
               lastResult.accepted, lastResult.threshold, lastResult.search_mode]);
  }
  return rows;
}

function csvCell(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

// ----------------------------------------------------------------- масштабируемость
function renderScalability(data) {
  const live = $('index-live');
  live.replaceChildren();
  const pairs = [
    ['Наблюдений в галерее', data.gallery_count],
    ['Быстрый поиск', data.ann.available
      ? 'доступен от ' + data.ann.min_items_for_ann + ' наблюдений' : 'недоступен'],
    ['Индекс', data.ann.kind],
  ];
  for (const [key, value] of pairs) {
    live.append(element('dt', '', key), element('dd', '', String(value)));
  }

  const target = $('scale-table');
  target.replaceChildren();
  if (!data.benchmark || !data.benchmark.rows || !data.benchmark.rows.length) {
    target.append(element('p', 'muted',
      'Замеров пока нет.'));
    $('scale-source').textContent = '';
    return;
  }
  const table = element('table');
  const head = element('thead');
  const headRow = element('tr');
  for (const title of ['Режим', 'мс/запрос', 'запросов/с', 'recall@10',
                       'верный ответ №1', 'память индекса', 'байт/вектор']) {
    headRow.append(element('th', '', title));
  }
  head.append(headRow);
  const body = element('tbody');
  for (const row of data.benchmark.rows) {
    const group = element('tr', 'group');
    const cell = element('td', '', 'Галерея ' + row.n.toLocaleString('ru-RU') +
      ' наблюдений, размерность ' + row.dim);
    cell.colSpan = 7;
    group.append(cell);
    body.append(group);
    for (const [name, mode] of Object.entries(row.modes)) {
      const line = element('tr');
      line.append(element('td', '', name + ' — ' + mode.description));
      line.append(element('td', '', mode.ms_per_query.toFixed(3)));
      line.append(element('td', '', Math.round(mode.qps).toLocaleString('ru-RU')));
      line.append(element('td', '', mode.recall_at_k.toFixed(3)));
      line.append(element('td', '', mode.true_match_at_1.toFixed(3)));
      line.append(element('td', '', (mode.index_bytes / 1048576).toFixed(0) + ' МиБ'));
      line.append(element('td', '', String(mode.bytes_per_vector)));
      body.append(line);
    }
  }
  table.append(head, body);
  target.append(table);
  $('scale-source').textContent = 'Замер на синтетических векторах, построенных по статистике ' +
    'векторов модели.';
}

async function renderAbout() {
  const model = await (await api('/v1/model')).json();
  const target = $('about-model');
  target.replaceChildren();
  const pairs = [
    ['Порог уверенности', Number(model.threshold).toFixed(2)],
    ['Размер вектора', model.dimension ?? '—'],
    ['Вычисления', model.device === 'cuda' ? 'GPU' : 'CPU'],
    ['Версия модели', String(model.model_sha256).slice(0, 12)],
  ];
  for (const [key, value] of pairs) {
    target.append(element('dt', '', key), element('dd', '', String(value)));
  }
}

// ----------------------------------------------------------------- события
const queryBox = boxPicker('query-file', 'query-preview', 'query-canvas-wrap', 'query-box');
const addBox = boxPicker('add-file', 'add-preview', 'add-canvas-wrap', 'add-box');

$('search-form').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    const bbox = queryBox.read();
    const file = $('query-file').files[0];
    const mode = document.querySelector('input[name=mode]:checked').value;
    const body = new FormData();
    body.append('image', file);
    body.append('bbox', JSON.stringify(bbox));
    const queryId = $('query-id').value.trim();
    if (queryId) body.append('query_id', queryId);

    clearResults();
    message('Поиск…');
    const answer = await (await api('/v1/search?mode=' + mode, { method: 'POST', body })).json();
    lastResult = answer;
    lastQuery = { file, bbox, id: queryId };

    $('result-empty').classList.add('hidden');
    $('export-buttons').classList.remove('hidden');
    const info = $('result-info');
    info.replaceChildren(element('strong', '', answer.accepted ? 'Совпадение найдено' : 'Совпадение не найдено'));
    const facts = element('div', 'facts');
    facts.append(element('span', '', 'порог ' + Number(answer.threshold).toFixed(2)));
    facts.append(element('span', '', 'галерея ' + answer.gallery_count));
    facts.append(element('span', '', 'режим ' + (answer.search_mode === 'ann' ? 'быстрый' : 'точный')));
    facts.append(element('span', '', Number(answer.seconds).toFixed(2) + ' с'));
    // Порядок уточняется по сходству кандидатов между собой, поэтому уверенность в списке
    // может идти не строго по убыванию. Говорим об этом, только когда так и вышло.
    const scores = answer.ranking.map(x => Number(x.cosine));
    if (scores.some((s, i) => i && s > scores[i - 1])) {
      facts.append(element('span', '', 'порядок учитывает сходство кандидатов между собой'));
    }
    info.append(facts);
    for (const note of answer.notes || []) info.append(element('small', '', note));
    info.className = 'result-status' + (answer.accepted ? '' : ' refused');

    answer.ranking.forEach(item => card(item, $('results'), { explainable: true, accepted: answer.accepted }));
    message('');
    await health();
  });
});

$('export-csv').addEventListener('click', () => {
  if (!lastResult) return;
  download('search_result.csv', exportRows().map(r => r.map(csvCell).join(',')).join('\n'),
           'text/csv;charset=utf-8');
});

$('export-json').addEventListener('click', () => {
  if (!lastResult) return;
  download('search_result.json', JSON.stringify(lastResult, null, 2), 'application/json');
});

$('add-form').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    const body = new FormData();
    body.append('image', $('add-file').files[0]);
    body.append('bbox', JSON.stringify(addBox.read()));
    if ($('add-id').value.trim()) body.append('image_id', $('add-id').value.trim());
    message('Добавление…');
    const added = await (await api('/v1/gallery/items', { method: 'POST', body })).json();
    clearResults();
    await loadGallery(); await health();
    message('Наблюдение добавлено.');
  });
});

$('import-form').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    const body = new FormData();
    body.append('archive', $('archive-file').files[0]);
    message('Импорт галереи…');
    const result = await (await api('/v1/gallery/import', { method: 'POST', body })).json();
    clearResults();
    await loadGallery(); await health();
    message('Импортировано: ' + result.added + ' ' + plural(result.added, 'наблюдение', 'наблюдения', 'наблюдений') + '.');
  });
});

for (const name of TABS) {
  $('tab-' + name).addEventListener('click', () => {
    for (const other of TABS) {
      $('tab-' + other).setAttribute('aria-selected', String(name === other));
      $(other + '-section').classList.toggle('hidden', name !== other);
    }
    if (name === 'gallery') run(loadGallery);
    if (name === 'about') run(async () => {
      await renderAbout();
      renderScalability(await (await api('/v1/index/stats')).json());
    });
  });
}

$('connect').addEventListener('click', () => run(async () => {
  apiKey = $('api-key').value;
  resetImageCache(); clearResults();
  await health(); await loadGallery();
  message('Подключение установлено.');
}));

$('refresh-gallery').addEventListener('click', () => run(async () => {
  await loadGallery(); await health();
  message('Галерея обновлена.');
}));

$('prev').addEventListener('click', () => run(async () => {
  offset = Math.max(0, offset - PAGE);
  await loadGallery();
}));

$('next').addEventListener('click', () => run(async () => {
  offset += PAGE;
  await loadGallery();
}));

// ----------------------------------------------------------------- демонстрация
/* Кнопки появляются только если демонстрационный набор реально лежит в сборке:
   обещать в интерфейсе то, чего нет в контейнере, хуже, чем не обещать вовсе.
   Но молчать тоже нельзя: набор собран из кадров организаторов и в репозиторий не
   выкладывается, поэтому вместо кнопок показываем команду, которой он собирается. */
async function setupDemo() {
  let info;
  try { info = await (await api('/v1/demo')).json(); } catch { return; }
  if (!info.available) {
    const hint = $('demo-hint');
    if (hint && info.hint) {
      hint.textContent = 'Демонстрационный набор не собран (' + (info.why || '') +
        '). Собрать одной командой: ' + info.hint;
      hint.classList.remove('hidden');
    }
    return;
  }

  $('use-demo').classList.remove('hidden');
  $('use-demo').addEventListener('click', () => run(async () => {
    const blob = await (await api(info.image_url)).blob();
    await queryBox.load(blob, 'demo-query.jpg', info.bbox);
    if (info.image_id) $('query-id').value = info.image_id;
    message('Демонстрационный кадр загружен, рамка выставлена. Нажмите «Найти автомобиль».');
  }));

  $('load-demo-gallery').classList.remove('hidden');
  $('load-demo-gallery').addEventListener('click', () => run(async () => {
    message('Импортирую демонстрационную галерею: ' + info.gallery_items + ' кадров…');
    const result = await (await api('/v1/demo/gallery', { method: 'POST' })).json();
    clearResults();
    await loadGallery(); await health();
    message('Демонстрационная галерея загружена: ' + result.added + ' наблюдений. ' + info.note);
  }));
}

window.addEventListener('beforeunload', resetImageCache);
run(async () => { await health(); await loadGallery(); await setupDemo(); });
