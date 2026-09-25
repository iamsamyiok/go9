/* engine.js — 9路迷你围棋:规则引擎 + 蒙特卡洛树搜索(浏览器版)
 * 与 Python(go_core.py / go_ai.py)逐条对应,一致性由 vectors.json 测试保障
 * 在 Worker 中运行: onnxruntime-web 推理先验,MCTS 全程后台不卡 UI */
'use strict';

const N = 9, EMPTY = 0, BLACK = 1, WHITE = 2, PASS = N * N;
const OPP = { 1: 2, 2: 1 };

const NEIGH = [];
for (let i = 0; i < N * N; i++) {
  const x = i % N, y = (i / N) | 0, a = [];
  if (x + 1 < N) a.push(i + 1);
  if (x - 1 >= 0) a.push(i - 1);
  if (y + 1 < N) a.push(i + N);
  if (y - 1 >= 0) a.push(i - N);
  NEIGH.push(a);
}

function group(board, i) {
  const color = board[i];
  const stones = [i], libs = new Set(), seen = new Set([i]);
  while (stones.length) {
    const s = stones.pop();
    for (const n of NEIGH[s]) {
      if (board[n] === EMPTY) libs.add(n);
      else if (board[n] === color && !seen.has(n)) { seen.add(n); stones.push(n); }
    }
  }
  return { stones: seen, libs };
}

function tryPlay(board, i, color) {
  if (board[i] !== EMPTY) return null;
  const b = board.slice();
  b[i] = color;
  const opp = OPP[color];
  let cap = 0;
  for (const n of NEIGH[i]) {
    if (b[n] === opp) {
      const g = group(b, n);
      if (g.libs.size === 0) for (const s of g.stones) b[s] = EMPTY, cap++;
    }
  }
  if (group(b, i).libs.size === 0) return null;   // 禁入点(自杀)
  return { board: b, cap };
}

function legalMoves(board, color, history) {       // history: Set<boardKey>
  const out = [];
  for (let i = 0; i < N * N; i++) {
    if (board[i] !== EMPTY) continue;
    const r = tryPlay(board, i, color);
    if (r && !history.has(r.board.join(''))) out.push({ idx: i, board: r.board });
  }
  return out;
}

function score(board) {                            // 数子法: 活子+独占空点
  const own = new Array(N * N).fill(0);
  let sb = 0, sw = 0;
  for (const v of board) { if (v === BLACK) sb++; else if (v === WHITE) sw++; }
  const done = new Set();
  for (let i = 0; i < N * N; i++) {
    if (board[i] !== EMPTY || done.has(i)) continue;
    const region = [i], touch = new Set(), seen = new Set([i]);
    while (region.length) {
      const s = region.pop();
      for (const n of NEIGH[s]) {
        if (board[n] === EMPTY) { if (!seen.has(n)) { seen.add(n); region.push(n); } }
        else touch.add(board[n]);
      }
    }
    for (const r of seen) done.add(r);
    if (touch.size === 1) {
      const c = touch.values().next().value;
      if (c === BLACK) { sb += seen.size; for (const r of seen) own[r] = BLACK; }
      else if (c === WHITE) { sw += seen.size; for (const r of seen) own[r] = WHITE; }
    }
  }
  return { sb, sw, own };
}

function scoreZ(board) {
  const { sb, sw } = score(board);
  return sb > sw ? 1 : (sw > sb ? -1 : 0);
}

/* ---------------- 快速走子 ---------------- */
const ROLLOUT_PASS_PROB = 0.05, ROLLOUT_CAP = 14, ROLLOUT_MAX_PLY = 126;

function rolloutMoveProper(board, color) {
  if (Math.random() < ROLLOUT_PASS_PROB) return null;
  const idx = [];
  for (let i = 0; i < N * N; i++) if (board[i] === EMPTY) idx.push(i);
  for (let k = idx.length - 1; k > 0; k--) { const j = (Math.random() * (k + 1)) | 0; [idx[k], idx[j]] = [idx[j], idx[k]]; }
  for (let k = 0; k < idx.length && k < ROLLOUT_CAP; k++) {
    const r = tryPlay(board, idx[k], color);
    if (r) return r.board;
  }
  for (const i of idx) { const r = tryPlay(board, i, color); if (r) return r.board; }
  return null;
}

function rollout(board, color, passes) {
  let b = board, c = color, p = passes;
  for (let t = 0; t < ROLLOUT_MAX_PLY; t++) {
    if (p >= 2) break;
    const nb = rolloutMoveProper(b, c);
    if (nb === null) p++; else { b = nb; p = 0; }
    c = OPP[c];
  }
  return scoreZ(b);
}

/* ---------------- 蒙特卡洛树搜索 ---------------- */
const C_PUCT = 1.3, EARLY_PASS_PLY = 18;

class MCTSNode {
  constructor(board, color, history, passes, prior) {
    this.board = board; this.color = color; this.history = history;
    this.passes = passes; this.prior = prior;
    this.children = []; this.n = 0; this.w = 0; this.expanded = false;
  }
  get q() { return this.n ? this.w / this.n : 0; }
}

function priorsOf(session, board, color, legal) {
  // 输入平面: 黑/白/行棋方
  const x = new Float32Array(3 * N * N);
  for (let i = 0; i < N * N; i++) {
    if (board[i] === BLACK) x[i] = 1;
    else if (board[i] === WHITE) x[N * N + i] = 1;
  }
  if (color === WHITE) x.fill(1, 2 * N * N);
  const t = new ort.Tensor('float32', x, [1, 3, N, N]);
  return session.run({ board: t }).then(out => {
    const logits = out.logits.data;
    let max = -Infinity;
    const masked = new Float32Array(N * N + 1);
    for (let i = 0; i < N * N + 1; i++) masked[i] = -1e9;
    for (const m of legal) masked[m.idx] = 0;
    masked[PASS] = 0;
    for (let i = 0; i < N * N + 1; i++) if (masked[i] === 0 && logits[i] > max) max = logits[i];
    let s = 0;
    for (let i = 0; i < N * N + 1; i++) { const e = masked[i] === 0 ? Math.exp(logits[i] - max) : 0; masked[i] = e; s += e; }
    const pr = {};
    for (const m of legal) pr[m.idx] = masked[m.idx] / s;
    pr[PASS] = masked[PASS] / s;
    return pr;
  });
}

async function expand(node, session, root) {
  const legal = legalMoves(node.board, node.color, node.history);
  const pr = await priorsOf(session, node.board, node.color, legal);
  if (root) {
    pr[PASS] *= 0.4;
    let s = 0; for (const k in pr) s += pr[k];
    for (const k in pr) pr[k] /= (s || 1);
  }
  for (const m of legal)
    node.children.push(new MCTSNode(m.board, OPP[node.color], new Set([...node.history, m.board.join('')]), 0, pr[m.idx]));
  node.children.push(new MCTSNode(node.board, OPP[node.color], node.history, node.passes + 1, pr[PASS]));
  node.expanded = true;
}

function pick(node) {
  let best = null, bs = -Infinity;
  const sq = Math.sqrt(node.n) || 1;
  for (const ch of node.children) {
    const u = ch.q + C_PUCT * ch.prior * sq / (1 + ch.n);
    if (u > bs) { bs = u; best = ch; }
  }
  return best;
}

async function mctsSearch(session, board, color, history, passes, sims, ply, onProgress, cancelled) {
  const root = new MCTSNode(board, color, history, passes, 0);
  await expand(root, session, true);
  let it = 0;
  for (; it < sims; it++) {
    if ((it & 15) === 0) {
      if (cancelled && cancelled()) break;
      if (onProgress) onProgress(it);
      /* 让出事件循环:主线程模式下 UI 保持响应 */
      await new Promise(r => setTimeout(r, 0));
    }
    const path = [root];
    let node = root;
    while (node.expanded && node.passes < 2) {
      node = pick(node);
      path.push(node);
      if (node.n === 0) break;
    }
    let z;
    if (node.passes >= 2) z = scoreZ(node.board);
    else {
      if (!node.expanded) await expand(node, session, false);
      const child = pick(node);
      if (!child) z = scoreZ(node.board);
      else { path.push(child); z = rollout(child.board, child.color, child.passes); }
    }
    for (const nd of path) { nd.n++; nd.w += (nd.color === BLACK ? z : -z); }
  }
  let best = root.children.reduce((a, b) => (b.n > a.n ? b : a));
  if (best.passes > passes && ply < EARLY_PASS_PLY) {
    const stones = root.children.filter(c => c.passes === passes && c.n > 0);
    if (stones.length) best = stones.reduce((a, b) => (b.n > a.n ? b : a));
  }
  const wr = (1 - best.q) / 2;
  const idx = best.passes > passes ? PASS : best.board.findIndex((v, i) => v !== board[i]);
  return { idx, wr, iters: it };
}

/* ---------------- 导出(node 测试用) ---------------- */
if (typeof module !== 'undefined') {
  module.exports = { N, EMPTY, BLACK, WHITE, PASS, NEIGH, group, tryPlay, legalMoves, score, scoreZ };
}
