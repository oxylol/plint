/* Counter-based hashing for everything the view draws "at random" (chip orientations without a
 * velocity, spray drops): a value is a pure function of its integer inputs, so the same time and data
 * always draw the same picture and nothing keeps random state (the rule of docs/architecture.md 9).
 *
 * lowbias32 (C. Wellons, "Prospecting for hash functions", 2018: the 32-bit integer finaliser with
 * the lowest measured avalanche bias of its search), combined over the inputs with the boost
 * hash_combine step. Uniform in [0, 1) with 2^-32 resolution. */

function lowbias32(value) {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

/** 32-bit hash of a stream name (FNV-1a over UTF-16 code units). */
export function streamKey(name) {
  let h = 0x811c9dc5;
  const text = String(name);
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** 32-bit hash of integers (stream key first, then counters). */
export function hash32(...values) {
  let h = 0x9e3779b9;
  for (const value of values) {
    const v = lowbias32((value | 0) ^ Math.floor(value / 4294967296));
    h = lowbias32(h ^ (v + 0x9e3779b9 + (h << 6) + (h >>> 2)));
  }
  return h >>> 0;
}

/** Uniform in [0, 1) from a stream name and integer counters. */
export function hashUniform(stream, ...counters) {
  return hash32(streamKey(stream), ...counters) / 4294967296;
}
