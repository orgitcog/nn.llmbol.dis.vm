/**
 * Inference Engine
 *
 * A simplified but functionally correct transformer-style inference engine.
 * Supports embedding lookup, multi-head self-attention (with KV-cache),
 * feed-forward networks with GELU, layer normalisation, top-K sampling,
 * and autoregressive text generation.
 */

import type { Tensor, MLModel } from './ml.m';
import { MLModule } from './ml.m';

export interface InferenceConfig {
  batchSize: number;
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
}

/**
 * Pre-computed embedding weight matrix.
 * `weights` is row-major with shape [vocabSize × hiddenSize].
 */
export interface EmbeddingWeights {
  vocabSize: number;
  hiddenSize: number;
  weights: Float32Array;
}

export class InferenceEngine {
  private _mlModule: MLModule;
  private _config: InferenceConfig;

  /** Lazily-initialised weight matrices, keyed by a descriptive string. */
  private _weights: Map<string, Float32Array>;

  constructor(config: Partial<InferenceConfig> = {}) {
    this._mlModule = new MLModule();
    this._config = {
      batchSize: config.batchSize ?? 1,
      maxTokens: config.maxTokens ?? 512,
      temperature: config.temperature ?? 1.0,
      topK: config.topK ?? 50,
      topP: config.topP ?? 0.95,
    };
    this._weights = new Map();
  }

  /*
   * ---------------------------------------------------------------------------
   * Public API
   * ---------------------------------------------------------------------------
   */

  /**
   * Run a single forward pass and return predicted token ids (one per input position).
   */
  async infer(model: MLModel, inputTokens: number[]): Promise<number[]> {
    const inputTensor = this._tokensToTensor(inputTokens);
    const logits = await this._forward(model, inputTensor);

    return this._tensorToTokens(logits);
  }

  /**
   * Autoregressively generate up to `maxTokens` new tokens given a prompt.
   * Uses a per-layer KV-cache to avoid redundant computation.
   */
  async generate(model: MLModel, prompt: number[], maxTokens: number = this._config.maxTokens): Promise<number[]> {
    const output = [...prompt];

    // KV-cache: layerIdx → { k, v } Float32Arrays accumulating all past tokens.
    const kvCache = new Map<number, { k: Float32Array; v: Float32Array }>();

    // Warm the cache with the full prompt
    const promptTensor = this._tokensToTensor(output);
    await this._forward(model, promptTensor, kvCache);

    for (let i = 0; i < maxTokens; i++) {
      // Feed only the most-recent token; KV-cache supplies the history
      const stepTensor = this._tokensToTensor([output[output.length - 1]]);
      const logits = await this._forward(model, stepTensor, kvCache);
      const nextToken = this._sample(logits);
      output.push(nextToken);

      if (nextToken === 0) {
        break; // EOS
      }
    }

    return output;
  }

  /**
   * Return inference configuration and basic statistics.
   */
  getStats() {
    return {
      config: this._config,
      cachedWeightCount: this._weights.size,
    };
  }

  /*
   * ---------------------------------------------------------------------------
   * Embedding helpers
   * ---------------------------------------------------------------------------
   */

  /**
   * Create a random embedding weight matrix for the given vocabulary and model
   * dimension.  Uses Xavier-style scaling.
   */
  createEmbeddings(vocabSize: number, hiddenSize: number): EmbeddingWeights {
    const size = vocabSize * hiddenSize;
    const weights = new Float32Array(size);
    const scale = Math.sqrt(1.0 / hiddenSize);

    for (let i = 0; i < size; i++) {
      weights[i] = (Math.random() * 2 - 1) * scale;
    }

    return { vocabSize, hiddenSize, weights };
  }

  /*
   * ---------------------------------------------------------------------------
   * Private transformer building blocks
   * ---------------------------------------------------------------------------
   */

  /**
   * Look up embedding vectors for an array of token ids.
   * Returns a tensor of shape [seqLen, hiddenSize].
   */
  private _embed(tokens: number[], embWeights: EmbeddingWeights): Tensor {
    const { hiddenSize, vocabSize, weights } = embWeights;
    const seqLen = tokens.length;
    const data = new Float32Array(seqLen * hiddenSize);

    for (let i = 0; i < seqLen; i++) {
      const tokenId = Math.max(0, Math.min(vocabSize - 1, Math.floor(tokens[i])));
      const srcOff = tokenId * hiddenSize;
      const dstOff = i * hiddenSize;

      for (let d = 0; d < hiddenSize; d++) {
        data[dstOff + d] = weights[srcOff + d];
      }
    }

    return { shape: [seqLen, hiddenSize], data, dtype: 'float32' };
  }

  /**
   * Layer normalisation over the last dimension (per-token).
   * Output = (x − mean) / sqrt(var + eps)
   */
  private _layerNorm(tensor: Tensor, eps: number = 1e-5): Tensor {
    const data = tensor.data as Float32Array;
    const shape = tensor.shape;
    const hiddenSize = shape[shape.length - 1];
    const seqLen = data.length / hiddenSize;
    const result = new Float32Array(data.length);

    for (let i = 0; i < seqLen; i++) {
      const off = i * hiddenSize;
      let mean = 0;

      for (let d = 0; d < hiddenSize; d++) {
        mean += data[off + d];
      }

      mean /= hiddenSize;

      let variance = 0;

      for (let d = 0; d < hiddenSize; d++) {
        const diff = data[off + d] - mean;
        variance += diff * diff;
      }

      variance /= hiddenSize;

      const invStd = 1.0 / Math.sqrt(variance + eps);

      for (let d = 0; d < hiddenSize; d++) {
        result[off + d] = (data[off + d] - mean) * invStd;
      }
    }

    return { shape: [...shape], data: result, dtype: 'float32' };
  }

  /**
   * Scaled dot-product attention.
   *
   * Q shape: [queryLen, headDim]
   * K shape: [keyLen,   headDim]
   * V shape: [keyLen,   headDim]
   * Output:  [queryLen, headDim]
   */
  private _scaledDotProductAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    queryLen: number,
    keyLen: number,
    headDim: number,
  ): Tensor {
    const qd = q.data as Float32Array;
    const kd = k.data as Float32Array;
    const vd = v.data as Float32Array;
    const scale = 1.0 / Math.sqrt(headDim);

    // scores = Q @ K^T / sqrt(headDim) — shape [queryLen, keyLen]
    const scores = new Float32Array(queryLen * keyLen);

    for (let i = 0; i < queryLen; i++) {
      for (let j = 0; j < keyLen; j++) {
        let dot = 0;

        for (let d = 0; d < headDim; d++) {
          dot += qd[i * headDim + d] * kd[j * headDim + d];
        }

        scores[i * keyLen + j] = dot * scale;
      }
    }

    // Row-wise softmax
    for (let i = 0; i < queryLen; i++) {
      let rowMax = scores[i * keyLen];

      for (let j = 1; j < keyLen; j++) {
        if (scores[i * keyLen + j] > rowMax) {
          rowMax = scores[i * keyLen + j];
        }
      }

      let expSum = 0;

      for (let j = 0; j < keyLen; j++) {
        scores[i * keyLen + j] = Math.exp(scores[i * keyLen + j] - rowMax);
        expSum += scores[i * keyLen + j];
      }

      for (let j = 0; j < keyLen; j++) {
        scores[i * keyLen + j] /= expSum;
      }
    }

    // output = softmax(scores) @ V — shape [queryLen, headDim]
    const output = new Float32Array(queryLen * headDim);

    for (let i = 0; i < queryLen; i++) {
      for (let d = 0; d < headDim; d++) {
        let val = 0;

        for (let j = 0; j < keyLen; j++) {
          val += scores[i * keyLen + j] * vd[j * headDim + d];
        }

        output[i * headDim + d] = val;
      }
    }

    return { shape: [queryLen, headDim], data: output, dtype: 'float32' };
  }

  /**
   * Multi-head attention.
   *
   * @param x         Input tensor [seqLen, hiddenSize].
   * @param seqLen    Number of query tokens (1 when using KV-cache).
   * @param hiddenSize Model dimension.
   * @param numHeads  Number of attention heads.
   * @param layerIdx  Used to select per-layer weight matrices.
   * @param kvCache   Optional KV-cache; updated in-place.
   */
  private _multiHeadAttention(
    x: Tensor,
    seqLen: number,
    hiddenSize: number,
    numHeads: number,
    layerIdx: number,
    kvCache?: Map<number, { k: Float32Array; v: Float32Array }>,
  ): Tensor {
    const headDim = Math.floor(hiddenSize / numHeads);
    const xd = x.data as Float32Array;

    /*
     * Compute Q, K, V projections per head — stored flat as
     * [numHeads × seqLen × headDim]
     */
    const newQ = new Float32Array(numHeads * seqLen * headDim);
    const newK = new Float32Array(numHeads * seqLen * headDim);
    const newV = new Float32Array(numHeads * seqLen * headDim);

    for (let h = 0; h < numHeads; h++) {
      const wq = this._getWeight(`L${layerIdx}_wq_h${h}`, hiddenSize, headDim);
      const wk = this._getWeight(`L${layerIdx}_wk_h${h}`, hiddenSize, headDim);
      const wv = this._getWeight(`L${layerIdx}_wv_h${h}`, hiddenSize, headDim);
      const base = h * seqLen * headDim;

      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < headDim; j++) {
          let qv = 0;
          let kv = 0;
          let vv = 0;

          for (let d = 0; d < hiddenSize; d++) {
            const xi = xd[i * hiddenSize + d];
            const wIdx = d * headDim + j;
            qv += xi * wq[wIdx];
            kv += xi * wk[wIdx];
            vv += xi * wv[wIdx];
          }

          newQ[base + i * headDim + j] = qv;
          newK[base + i * headDim + j] = kv;
          newV[base + i * headDim + j] = vv;
        }
      }
    }

    // Merge with KV-cache if available
    let totalKeyLen = seqLen;
    let allK = newK;
    let allV = newV;

    if (kvCache !== undefined) {
      const cached = kvCache.get(layerIdx);

      if (cached !== undefined) {
        const cachedKeyLen = Math.floor(cached.k.length / (numHeads * headDim));
        totalKeyLen = cachedKeyLen + seqLen;

        allK = new Float32Array(numHeads * totalKeyLen * headDim);
        allV = new Float32Array(numHeads * totalKeyLen * headDim);

        for (let h = 0; h < numHeads; h++) {
          const srcBase = h * cachedKeyLen * headDim;
          const dstBase = h * totalKeyLen * headDim;

          // Copy cached tokens
          allK.set(cached.k.subarray(srcBase, srcBase + cachedKeyLen * headDim), dstBase);
          allV.set(cached.v.subarray(srcBase, srcBase + cachedKeyLen * headDim), dstBase);

          // Append new tokens
          const newBase = h * seqLen * headDim;
          allK.set(newK.subarray(newBase, newBase + seqLen * headDim), dstBase + cachedKeyLen * headDim);
          allV.set(newV.subarray(newBase, newBase + seqLen * headDim), dstBase + cachedKeyLen * headDim);
        }
      }

      kvCache.set(layerIdx, { k: allK, v: allV });
    }

    // Per-head attention → concatenate into [seqLen, hiddenSize]
    const headOutputs = new Float32Array(seqLen * hiddenSize);

    for (let h = 0; h < numHeads; h++) {
      const qSlice = newQ.subarray(h * seqLen * headDim, (h + 1) * seqLen * headDim);
      const kSlice = allK.subarray(h * totalKeyLen * headDim, (h + 1) * totalKeyLen * headDim);
      const vSlice = allV.subarray(h * totalKeyLen * headDim, (h + 1) * totalKeyLen * headDim);

      const qT: Tensor = { shape: [seqLen, headDim], data: qSlice, dtype: 'float32' };
      const kT: Tensor = { shape: [totalKeyLen, headDim], data: kSlice, dtype: 'float32' };
      const vT: Tensor = { shape: [totalKeyLen, headDim], data: vSlice, dtype: 'float32' };

      const attnOut = this._scaledDotProductAttention(qT, kT, vT, seqLen, totalKeyLen, headDim);
      const aod = attnOut.data as Float32Array;

      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < headDim; j++) {
          headOutputs[i * hiddenSize + h * headDim + j] = aod[i * headDim + j];
        }
      }
    }

    // Output projection W_O: [hiddenSize, hiddenSize]
    const wo = this._getWeight(`L${layerIdx}_wo`, hiddenSize, hiddenSize);
    const projected = new Float32Array(seqLen * hiddenSize);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        let val = 0;

        for (let d = 0; d < hiddenSize; d++) {
          val += headOutputs[i * hiddenSize + d] * wo[d * hiddenSize + j];
        }

        projected[i * hiddenSize + j] = val;
      }
    }

    return { shape: [seqLen, hiddenSize], data: projected, dtype: 'float32' };
  }

  /**
   * Two-layer feed-forward network with GELU activation.
   *
   * x → Linear(hiddenSize → ffnSize) → GELU → Linear(ffnSize → hiddenSize)
   */
  private _feedForward(x: Tensor, hiddenSize: number, ffnSize: number, layerIdx: number): Tensor {
    const xd = x.data as Float32Array;
    const seqLen = x.shape[0];

    const w1 = this._getWeight(`L${layerIdx}_ffn_w1`, hiddenSize, ffnSize);
    const w2 = this._getWeight(`L${layerIdx}_ffn_w2`, ffnSize, hiddenSize);

    // Hidden layer with GELU
    const hidden = new Float32Array(seqLen * ffnSize);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < ffnSize; j++) {
        let val = 0;

        for (let d = 0; d < hiddenSize; d++) {
          val += xd[i * hiddenSize + d] * w1[d * ffnSize + j];
        }

        hidden[i * ffnSize + j] = this._gelu(val);
      }
    }

    // Output projection
    const out = new Float32Array(seqLen * hiddenSize);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        let val = 0;

        for (let d = 0; d < ffnSize; d++) {
          val += hidden[i * ffnSize + d] * w2[d * hiddenSize + j];
        }

        out[i * hiddenSize + j] = val;
      }
    }

    return { shape: [seqLen, hiddenSize], data: out, dtype: 'float32' };
  }

  /**
   * Single transformer layer: pre-norm attention + residual, then pre-norm FFN + residual.
   */
  private _processLayer(
    hidden: Tensor,
    seqLen: number,
    hiddenSize: number,
    numHeads: number,
    layerIdx: number,
    kvCache?: Map<number, { k: Float32Array; v: Float32Array }>,
  ): Tensor {
    // --- Attention sub-layer ---
    const residual1 = this._mlModule.clone(hidden);
    const normed1 = this._layerNorm(hidden);
    const attnOut = this._multiHeadAttention(normed1, seqLen, hiddenSize, numHeads, layerIdx, kvCache);
    const afterAttn = this._mlModule.add(attnOut, residual1);

    // --- FFN sub-layer ---
    const residual2 = this._mlModule.clone(afterAttn);
    const normed2 = this._layerNorm(afterAttn);
    const ffnOut = this._feedForward(normed2, hiddenSize, 4 * hiddenSize, layerIdx);
    const afterFfn = this._mlModule.add(ffnOut, residual2);

    return afterFfn;
  }

  /**
   * Full forward pass: embed → N layers → final layer norm → vocab projection.
   *
   * Returns a flat logits tensor of shape [vocabSize] for the *last* token
   * position (suitable for next-token prediction).
   */
  private async _forward(
    model: MLModel,
    input: Tensor,
    kvCache?: Map<number, { k: Float32Array; v: Float32Array }>,
  ): Promise<Tensor> {
    const { vocabSize, hiddenSize, numLayers, numHeads } = model.config;
    const inputData = input.data as Float32Array;
    const seqLen = inputData.length;

    // Embedding lookup using lazily-created weights
    const embWeights: EmbeddingWeights = {
      vocabSize,
      hiddenSize,
      weights: this._getWeight('emb', vocabSize, hiddenSize),
    };

    let hidden = this._embed(Array.from(inputData).map(Math.floor), embWeights);

    // Transformer layers
    for (let l = 0; l < numLayers; l++) {
      hidden = this._processLayer(hidden, seqLen, hiddenSize, numHeads, l, kvCache);
    }

    // Final layer norm
    hidden = this._layerNorm(hidden);

    // Project last token position to vocab logits
    const wLmHead = this._getWeight('lm_head', hiddenSize, vocabSize);
    const hiddenData = hidden.data as Float32Array;
    const lastOff = (seqLen - 1) * hiddenSize;
    const logits = new Float32Array(vocabSize);

    for (let v = 0; v < vocabSize; v++) {
      let val = 0;

      for (let d = 0; d < hiddenSize; d++) {
        val += hiddenData[lastOff + d] * wLmHead[d * vocabSize + v];
      }

      logits[v] = val;
    }

    return { shape: [vocabSize], data: logits, dtype: 'float32' };
  }

  /**
   * Sample the next token from `logits` using top-K filtering + temperature scaling.
   *
   * Top-K: keep only the K highest-scoring tokens, set others to -Infinity,
   * then apply softmax and sample from the resulting distribution.
   */
  private _sample(logits: Tensor): number {
    const data = logits.data as Float32Array;
    const n = data.length;
    const topK = Math.min(this._config.topK, n);

    // Apply temperature
    const scaled = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      scaled[i] = data[i] / (this._config.temperature || 1);
    }

    // Find the K-th largest value (partial sort via selection)
    const sorted = Float32Array.from(scaled).sort().reverse();
    const threshold = sorted[topK - 1];

    // Mask values below threshold
    for (let i = 0; i < n; i++) {
      if (scaled[i] < threshold) {
        scaled[i] = -Infinity;
      }
    }

    // Softmax
    let maxVal = scaled[0];

    for (let i = 1; i < n; i++) {
      if (scaled[i] > maxVal) {
        maxVal = scaled[i];
      }
    }

    let expSum = 0;
    const probs = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      probs[i] = Math.exp(scaled[i] - maxVal);
      expSum += probs[i];
    }

    for (let i = 0; i < n; i++) {
      probs[i] /= expSum;
    }

    // Sample
    let r = Math.random();

    for (let i = 0; i < n; i++) {
      r -= probs[i];

      if (r <= 0) {
        return i;
      }
    }

    return n - 1;
  }

  /*
   * ---------------------------------------------------------------------------
   * Utilities
   * ---------------------------------------------------------------------------
   */

  /** Pack token ids into a flat float32 tensor of shape [seqLen]. */
  private _tokensToTensor(tokens: number[]): Tensor {
    return {
      shape: [tokens.length],
      data: Float32Array.from(tokens),
      dtype: 'float32',
    };
  }

  /**
   * Convert a logits tensor [vocabSize] to a token prediction array via argmax.
   * When given a full-sequence logits tensor it returns one token per position.
   */
  private _tensorToTokens(logits: Tensor): number[] {
    const data = logits.data as Float32Array;
    const shape = logits.shape;

    // Shape [vocabSize] — single position, return argmax
    if (shape.length === 1) {
      let best = 0;

      for (let i = 1; i < data.length; i++) {
        if (data[i] > data[best]) {
          best = i;
        }
      }

      return [best];
    }

    // Shape [seqLen, vocabSize] — argmax per position
    const [seqLen, vocabSize] = shape;
    const tokens: number[] = [];

    for (let i = 0; i < seqLen; i++) {
      let best = 0;

      for (let v = 1; v < vocabSize; v++) {
        if (data[i * vocabSize + v] > data[i * vocabSize + best]) {
          best = v;
        }
      }

      tokens.push(best);
    }

    return tokens;
  }

  /**
   * Gaussian Error Linear Unit (GELU) activation.
   * Uses the tanh approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   */
  private _gelu(x: number): number {
    const c = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
  }

  /**
   * Return (lazily creating) a weight matrix stored under `key`.
   * Uses a deterministic LCG seeded from the key string so the same weights
   * are returned on every call with the same key.
   */
  private _getWeight(key: string, rows: number, cols: number): Float32Array {
    if (!this._weights.has(key)) {
      const size = rows * cols;
      const w = new Float32Array(size);
      const scale = Math.sqrt(2.0 / (rows + cols));

      // LCG seeded from key for reproducibility
      let seed = 1337;

      for (let ci = 0; ci < key.length; ci++) {
        seed = (seed * 31 + key.charCodeAt(ci)) >>> 0;
      }

      for (let i = 0; i < size; i++) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        w[i] = ((seed / 0x100000000) * 2 - 1) * scale;
      }

      this._weights.set(key, w);
    }

    return this._weights.get(key)!;
  }
}

/**
 * Create a new InferenceEngine with optional configuration overrides.
 */
export function createInferenceEngine(config?: Partial<InferenceConfig>): InferenceEngine {
  return new InferenceEngine(config);
}
