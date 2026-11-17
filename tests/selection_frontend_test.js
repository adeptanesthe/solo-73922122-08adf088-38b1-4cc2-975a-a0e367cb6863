// 送检选样页前端逻辑测试：DOM 打桩 + 可控 fetch，验证
//  1) 启动初始化（10 条空待检行、原料名写入表头）
//  2) 载入演示样例后表格行数正确
//  3) 正常计算 → 推荐面板渲染（选中配方、增量叶、收益）
//  4) 无增量结果 → “不推荐送检”面板
//  5) pending → 待定面板
//  6) invalid → 错误面板
//  7) 计算期间修改 → 旧建议立即撤下；迟到响应不恢复（epoch 机制）
//  8) 撤下后重新计算仍可正常渲染
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'select.html'), 'utf8');
const pageJs = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ---- DOM 打桩 ----
const store = {};
function makeEl(id) {
  const e = {
    id, checked: false, textContent: '', innerHTML: '',
    style: { display: 'none' }, dataset: {}, disabled: false,
    listeners: {},
    classList: { add() {}, remove() {} },
    addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
    appendChild() {},
    querySelector() { return makeEl('q-' + Math.random()); },
    querySelectorAll() { return []; },
  };
  let v = '';
  Object.defineProperty(e, 'value', { get: () => v, set: x => { v = String(x); } });
  return e;
}
function el(id) { if (!store[id]) store[id] = makeEl(id); return store[id]; }

// 表格 tr/input 打桩：renderTable 向 tbody 追加行并绑定 input 事件
function makeInput(col) {
  const inp = { dataset: { col: String(col) }, listeners: {}, value: '' };
  inp.addEventListener = (ev, fn) => { (inp.listeners[ev] ??= []).push(fn); };
  return inp;
}
function makeRow(vals) {
  const inputs = vals.map((v, c) => makeInput(c));
  inputs.forEach((inp, c) => { inp.value = vals[c] === null ? '' : String(vals[c]); });
  const tr = {
    dataset: {},
    querySelectorAll: sel => sel === 'input' ? inputs : [],
    querySelector: () => ({ addEventListener: (ev, fn) => { tr._del = fn; } }),
  };
  return tr;
}
const tbodyStore = { testedTable: [], candTable: [] };
function makeTbody(name) {
  return {
    set innerHTML(x) { if (x === '') tbodyStore[name] = []; },
    get innerHTML() { return ''; },
    appendChild(tr) { tbodyStore[name].push(tr); },
    querySelectorAll: () => tbodyStore[name],
  };
}
store.testedTable = makeTbody('testedTable');
store.candTable = makeTbody('candTable');

global.document = {
  getElementById: el,
  querySelector: sel => {
    if (sel === '#testedTable tbody') return store.testedTable;
    if (sel === '#candTable tbody') return store.candTable;
    return makeEl('q');
  },
  querySelectorAll: sel => sel.startsWith('.fn') ? [makeEl('th1'), makeEl('th2')] : [],
  createElement: () => makeRow([null, null, null, null]),
};
let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
global.fetch = (...a) => fetchImpl(...a);

// 页面 HTML 中名额输入框默认 value=3，打桩环境不解析 HTML，这里预置默认值
el('slots').value = '3';

const META = { feature_names: ['再生PP', '再生PE', '矿物填料', '功能助剂'], total_grams: 1000,
  proba_threshold: 0.7, n_trees: 5, max_depth: 3, tree_depths: [3, 3, 3, 3, 3], train_seed: 1 };
const USEFUL = {
  tested: [[250, 250, 350, 150], [260, 240, 350, 150]],
  candidates: Array.from({ length: 25 }, (_, i) => [300 + i, 200, 300 - i, 200]),
  slots: 3,
};
const NO_GAIN = {
  tested: [[100, 100, 100, 700]],
  candidates: Array.from({ length: 12 }, (_, i) => [110 + i, 90, 100, 700 - i]),
  slots: 3,
};
const SAMPLES = { selection: { useful: USEFUL, no_gain: NO_GAIN } };
const OK_RESULT = {
  status: 'ok', selected: [[301, 200, 299, 200], [305, 200, 295, 200]],
  selected_indices: [1, 5], benefit: 0.041234, benefit_exact: '123/2987',
  gain_leaves: [
    { tree: 0, node: 3, n_train: 999, weight: 0.001001 },
    { tree: 1, node: 7, n_train: 135, weight: 0.007407 },
  ],
  exhaustive: true, nodes_visited: 42, elapsed_ms: 12.3, reason: '已穷举证明最优',
};

fetchImpl = url => Promise.resolve({
  json: async () => url.includes('meta') ? META : SAMPLES,
});

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const assert = (name, cond) => { console.log((cond ? '[PASS] ' : '[FAIL] ') + name); if (!cond) process.exitCode = 1; };

  await sleep(20);
  assert('启动后生成 10 条待检行', tbodyStore.candTable.length === 10);
  assert('已检表初始为空', tbodyStore.testedTable.length === 0);
  assert('名额默认 3', document.getElementById('slots').value === '3');

  // --- 载入演示样例 ---
  loadSample(SAMPLES.selection.useful);
  await sleep(5);
  assert('载入样例后已检 2 行', tbodyStore.testedTable.length === 2);
  assert('载入样例后待检 25 行', tbodyStore.candTable.length === 25);
  assert('修改后状态提示已撤下', document.getElementById('status').textContent.includes('撤下'));

  // --- 正常计算渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await runSelect();
  assert('ok 面板显示', document.getElementById('panelOk').style.display === 'block');
  assert('送检条数渲染为 2', String(document.getElementById('okK').textContent) === '2');
  assert('增量叶数渲染为 2', String(document.getElementById('okLeafN').textContent) === '2');
  assert('收益已渲染', document.getElementById('okBen').textContent.startsWith('0.041234'));
  assert('精确分数已渲染', document.getElementById('okBenExact').textContent === '123/2987');
  assert('选中表格含两条配方', document.getElementById('okRows').innerHTML.includes('301'));
  assert('叶明细含训练样本数', document.getElementById('okLeaves').innerHTML.includes('999'));

  // --- 无增量 → 不推荐 ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'ok', selected: [], reason: '没有增量收益' }) });
  await runSelect();
  assert('不推荐面板显示', document.getElementById('panelNoRec').style.display === 'block');
  assert('ok 面板隐藏', document.getElementById('panelOk').style.display === 'none');

  // --- pending ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'pending', reason: '预算耗尽' }) });
  await runSelect();
  assert('待定面板显示', document.getElementById('panelPending').style.display === 'block');

  // --- 后端 invalid ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'invalid', reason: '名额非法' }) });
  await runSelect();
  assert('invalid 面板显示', document.getElementById('panelInvalid').style.display === 'block');

  // --- 计算期间修改 → 立即撤下；迟到结果不恢复 ---
  let resolveStale;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => OK_RESULT }); });
  const p = runSelect();
  await sleep(5);
  onUserEdit();
  assert('修改后 ok 面板立即撤下', document.getElementById('panelOk').style.display === 'none');
  assert('撤下状态提示', document.getElementById('status').textContent.includes('撤下'));
  resolveStale();
  await p;
  await sleep(5);
  assert('迟到结果未恢复面板', document.getElementById('panelOk').style.display === 'none');
  assert('迟到结果未改写状态', document.getElementById('status').textContent.includes('撤下'));

  // --- 撤下后重新计算正常 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await runSelect();
  assert('重新计算后面板恢复', document.getElementById('panelOk').style.display === 'block');

  // --- 前端本地校验：名额超界直接 invalid 面板，不发请求 ---
  let called = false;
  fetchImpl = () => { called = true; return Promise.resolve({ json: async () => OK_RESULT }); };
  document.getElementById('slots').value = '9';
  const epochBefore = epoch;
  await runSelect();
  assert('名额超界本地拦截', document.getElementById('panelInvalid').style.display === 'block' && called === false);
  assert('校验失败也推进 epoch，迟到响应不会复活', epoch === epochBefore + 1);
  document.getElementById('slots').value = '3';
})();
`;

eval(pageJs + driver);
