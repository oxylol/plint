/* Display units and number formatting: SI values in the units a machinist reads.
 *
 * The same rules as the report (cnctwin/report/fmt.py, docs/report.md section 4): lengths in mm
 * (deviations, wear, roughness in µm by their size), temperatures in °C, power in W or kW, energy in
 * J, kJ or kWh, speeds in rpm, feeds in mm/min, volumes in mm³ or cm³, mass in g; 4 significant
 * figures, plain between 1e-4 and 1e9, the minus sign U+2212, a dash for a missing value. Additions:
 * cutting speeds of 1 m/s and more in m/min, pressures by size (Pa, kPa, MPa), flows in l/min.
 */

export const DASH = '–';
export const MINUS = '−';
export const ZERO_DEGC_K = 273.15;
const RPM_PER_RAD_S = 60 / (2 * Math.PI);
const DEG_PER_RAD = 180 / Math.PI;

export function isNum(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

/** The value as a finite number, else null (text, null, NaN, booleans). */
export function asNumber(value) {
  if (value === null || value === undefined || typeof value === 'boolean') return null;
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function withMinus(text) {
  if (!text.startsWith('-')) return text;
  const rest = text.slice(1);
  return /[1-9]/.test(rest) ? MINUS + rest : rest;
}

/** Fixed decimals. */
export function fixed(value, decimals = 3) {
  const number = asNumber(value);
  if (number === null) return DASH;
  return withMinus(number.toFixed(Math.max(0, Math.min(20, decimals))));
}

/** `digits` significant figures: plain between 1e-4 and 1e9, else mantissa e exponent. Whole numbers
 * (counts, ticks) are written as they are unless `whole` is false (measured values keep their
 * trailing zeros: 62.00 µm). */
export function num(value, digits = 4, whole = true) {
  const number = asNumber(value);
  if (number === null) return DASH;
  if (whole && Number.isInteger(number) && Math.abs(number) < 1e15) return withMinus(String(number));
  if (number === 0) return '0';
  const magnitude = Math.abs(number);
  if (magnitude >= 1e9 || magnitude < 1e-4) {
    const [mantissa, exponent] = number.toExponential(digits - 1).split('e');
    return `${withMinus(mantissa)}e${withMinus(String(Number(exponent)))}`;
  }
  const exponent = Math.floor(Math.log10(magnitude));
  let decimals = Math.max(0, digits - 1 - exponent);
  const rounded = Math.abs(Number(number.toFixed(decimals)));
  if (rounded >= 10 ** (exponent + 1)) decimals = Math.max(0, decimals - 1);
  return fixed(number, decimals);
}

/** With an explicit sign (+0.012, −0.004). */
export function signed(value, digits = 4) {
  const number = asNumber(value);
  if (number === null) return DASH;
  const text = num(number, digits);
  return number > 0 ? `+${text}` : text;
}

export function pct(fraction, decimals = 1) {
  const number = asNumber(fraction);
  return number === null ? DASH : `${fixed(number * 100, decimals)} %`;
}

/** 2.320 s, 3 min 12.4 s, 2 h 05 min. */
export function duration(seconds) {
  const number = asNumber(seconds);
  if (number === null) return DASH;
  const sign = number < 0 ? MINUS : '';
  const s = Math.abs(number);
  if (s < 60) return `${sign}${fixed(s, s < 10 ? 3 : 2)} s`;
  if (s < 3600) {
    const minutes = Math.floor(s / 60);
    return `${sign}${minutes} min ${fixed(s - 60 * minutes, 1)} s`;
  }
  const hours = Math.floor(s / 3600);
  const minutes = Math.round((s - 3600 * hours) / 60);
  return `${sign}${hours} h ${String(minutes).padStart(2, '0')} min`;
}

/** A playback clock: 0.000 s .. 59.999 s, then m:ss.sss, then h:mm:ss. */
export function clock(seconds) {
  const number = asNumber(seconds);
  if (number === null) return DASH;
  const s = Math.max(0, number);
  if (s < 60) return `${s.toFixed(3)} s`;
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s - 3600 * hours) / 60);
  const rest = s - 3600 * hours - 60 * minutes;
  const ss = rest.toFixed(hours ? 0 : 3).padStart(hours ? 2 : 6, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${ss}` : `${minutes}:${ss}`;
}

export function bytes(count) {
  const number = asNumber(count);
  if (number === null) return DASH;
  if (number < 1024) return `${number} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = number / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
  return `${num(value, 3)} ${units[i]}`;
}

export function eur(value) {
  const number = asNumber(value);
  if (number === null) return DASH;
  return `${number !== 0 && Math.abs(number) < 1 ? num(number, 3) : fixed(number, 2)} EUR`;
}

// ------------------------------------------------------------------ display units of SI units

const unit = (label, factor = 1, offset = 0) => ({ label, factor, offset });
const LADDERS = {
  m: [[1e-3, unit('µm', 1e6)], [1, unit('mm', 1e3)], [Infinity, unit('m')]],
  W: [[1e3, unit('W')], [Infinity, unit('kW', 1e-3)]],
  J: [[1e4, unit('J')], [1e7, unit('kJ', 1e-3)], [Infinity, unit('kWh', 1 / 3.6e6)]],
  m3: [[1e-6, unit('mm³', 1e9)], [1e-3, unit('cm³', 1e6)], [Infinity, unit('l', 1e3)]],
  kg: [[1e-4, unit('mg', 1e6)], [1, unit('g', 1e3)], [Infinity, unit('kg')]],
  Pa: [[1e3, unit('Pa')], [1e6, unit('kPa', 1e-3)], [Infinity, unit('MPa', 1e-6)]],
  'm/s': [[1, unit('mm/min', 60000)], [Infinity, unit('m/min', 60)]],
  N: [[1e4, unit('N')], [Infinity, unit('kN', 1e-3)]],
};
const FIXED_UNITS = {
  K: unit('°C', 1, -ZERO_DEGC_K), dK: unit('K'), 'rad/s': unit('rpm', RPM_PER_RAD_S), rad: unit('°', DEG_PER_RAD),
  'm3/s': unit('l/min', 60000), 'kg/s': unit('g/s', 1000), frac: unit('%', 100), '-': unit(''), '': unit(''), '1': unit(''),
  Nm: unit('N·m'), 'N m': unit('N·m'), 'm/s2': unit('m/s²'), 'rad/s2': unit('rad/s²'), m2: unit('mm²', 1e6),
  'N/m': unit('N/µm', 1e-6), 'm/rev': unit('mm/rev', 1e3), 'm/tooth': unit('mm/tooth', 1e3), Hz: unit('Hz'), s: unit('s'),
  'kg/m3': unit('kg/m³'), EUR: unit('EUR'), ticks: unit('ticks'), count: unit(''),
};
const UNIT_ALIASES = { rad_s: 'rad/s', m_s: 'm/s', m_s2: 'm/s2', m3_s: 'm3/s', kg_s: 'kg/s', 'N·m': 'Nm', m_per_rev: 'm/rev' };

/** The display unit {label, factor, offset} of SI unit text `si` for values reaching `magnitude`. */
export function displayUnit(si, magnitude = null) {
  const key = UNIT_ALIASES[String(si || '').trim()] || String(si || '').trim();
  if (FIXED_UNITS[key]) return FIXED_UNITS[key];
  const ladder = LADDERS[key];
  if (ladder) {
    const size = isNum(magnitude) ? Math.abs(magnitude) : 0;
    for (const [limit, choice] of ladder) if (size < limit) return choice;
    return ladder[ladder.length - 1][1];
  }
  return unit(key);
}

/** The largest finite |value| of arrays / numbers (for choosing a display unit). */
export function magnitudeOf(...values) {
  let best = 0;
  const visit = (v) => {
    if (v === null || v === undefined) return;
    if (typeof v === 'number') { if (Number.isFinite(v) && Math.abs(v) > best) best = Math.abs(v); return; }
    if (ArrayBuffer.isView(v) || Array.isArray(v)) { for (const x of v) visit(x); }
  };
  values.forEach(visit);
  return best;
}

export function convert(value, display) {
  const number = asNumber(value);
  return number === null ? null : number * display.factor + display.offset;
}

/** "62.0 µm" from 6.2e-5 m (display unit chosen by the value's own size unless `magnitude` is given). */
export function formatSI(value, si, { magnitude = null, digits = 4 } = {}) {
  const number = asNumber(value);
  if (number === null) return DASH;
  const display = displayUnit(si, magnitude === null ? number : magnitude);
  const text = num(number * display.factor + display.offset, digits, false);
  return display.label ? `${text} ${display.label}` : text;
}

// ------------------------------------------------------------------ named quantities (key carries its unit)

const KEY_UNITS = [
  ['_m_per_rev', 'm/rev', 1], ['_m_per_tooth', 'm/tooth', 1], ['_N_per_m', 'N/m', 1], ['_m3_s', 'm3/s', 1],
  ['_kg_s', 'kg/s', 1], ['_kg_m3', 'kg/m3', 1], ['_rad_s2', 'rad/s2', 1], ['_rad_s', 'rad/s', 1], ['_m_s2', 'm/s2', 1],
  ['_m_s', 'm/s', 1], ['_mm_min', 'm/s', 1e-3 / 60], ['_m_min', 'm/s', 1 / 60], ['_W_m2K', 'W/(m² K)', 1],
  ['_W_mK', 'W/(m K)', 1], ['_J_kgK', 'J/(kg K)', 1], ['_Nm', 'Nm', 1], ['_m3', 'm3', 1], ['_mm3', 'm3', 1e-9],
  ['_m2', 'm2', 1], ['_kHz', 'Hz', 1e3], ['_Hz', 'Hz', 1], ['_MPa', 'Pa', 1e6], ['_kPa', 'Pa', 1e3], ['_bar', 'Pa', 1e5],
  ['_Pa', 'Pa', 1], ['_dK', 'dK', 1], ['_degC', 'K', 1, ZERO_DEGC_K], ['_K', 'K', 1], ['_kWh', 'J', 3.6e6],
  ['_kJ', 'J', 1e3], ['_J', 'J', 1], ['_kW', 'W', 1e3], ['_W', 'W', 1], ['_kN', 'N', 1e3], ['_N', 'N', 1],
  ['_kg', 'kg', 1], ['_g', 'kg', 1e-3], ['_urad', 'rad', 1e-6], ['_deg', 'rad', Math.PI / 180], ['_rad', 'rad', 1],
  ['_rpm', 'rad/s', 2 * Math.PI / 60], ['_fraction', 'frac', 1], ['_frac', 'frac', 1], ['_pct', 'frac', 0.01],
  ['_eur', 'EUR', 1], ['_ticks', 'ticks', 1], ['_count', 'count', 1], ['_mm', 'm', 1e-3], ['_um', 'm', 1e-6],
  ['_nm', 'm', 1e-9], ['_min', 's', 60], ['_ms', 's', 1e-3], ['_us', 's', 1e-6], ['_s', 's', 1], ['_m', 'm', 1],
  ['_l', 'm3', 1e-3], ['_Nm_rad', 'N m/rad', 1],
].sort((a, b) => b[0].length - a[0].length);
const PASS_THROUGH = ['_HB', '_HV', '_HRC', '_HRB', '_id', '_index'];

/** {stem, si, factor, offset} of a key whose last token is its unit (architecture 3), or null. */
export function parseKey(key) {
  const text = String(key);
  if (PASS_THROUGH.some((s) => text.endsWith(s))) return null;
  for (const [suffix, si, factor, offset = 0] of KEY_UNITS) {
    if (text.endsWith(suffix) && text.length > suffix.length) {
      return { stem: text.slice(0, -suffix.length), si, factor, offset };
    }
  }
  return null;
}

/** A named quantity in display units: ('deviation_m', 6.2e-5) -> {label: 'deviation', text: '62.00 µm'}. */
export function quantity(key, value) {
  const parsed = parseKey(key);
  const label = String(parsed ? parsed.stem : key).replace(/_/g, ' ');
  if (Array.isArray(value)) {
    const parts = value.map((v) => quantity(key, v).text);
    return { label, text: parts.join(', ') };
  }
  const number = asNumber(value);
  if (number === null) {
    return { label, text: typeof value === 'string' ? value : (typeof value === 'boolean' ? (value ? 'yes' : 'no') : DASH) };
  }
  if (!parsed || ['EUR', 'count', 'ticks'].includes(parsed.si)) {
    if (parsed && parsed.si === 'EUR') return { label, text: eur(number) };
    return { label, text: num(number) };
  }
  const si = number * parsed.factor + parsed.offset;
  return { label, text: formatSI(si, parsed.si) };
}

export function quantityText(key, value) {
  const { label, text } = quantity(key, value);
  return `${label} ${text}`;
}

/** Shorthands for result tables. */
export const fmt = {
  mm: (m, decimals = 3) => (asNumber(m) === null ? DASH : `${fixed(m * 1e3, decimals)} mm`),
  um: (m, decimals = 1) => (asNumber(m) === null ? DASH : `${fixed(m * 1e6, decimals)} µm`),
  signedUm: (m, decimals = 1) => (asNumber(m) === null ? DASH : `${m > 0 ? '+' : ''}${fixed(m * 1e6, decimals)} µm`),
  degC: (K, decimals = 1) => (asNumber(K) === null ? DASH : `${fixed(K - ZERO_DEGC_K, decimals)} °C`),
  rpm: (radS) => (asNumber(radS) === null ? DASH : `${fixed(Math.abs(radS) * RPM_PER_RAD_S, 0)} rpm`),
  mmMin: (mS) => (asNumber(mS) === null ? DASH : `${num(mS * 60000, 4)} mm/min`),
  kW: (W) => (asNumber(W) === null ? DASH : `${num(W / 1000, 4)} kW`),
  kJ: (J) => (asNumber(J) === null ? DASH : `${num(J / 1000, 4)} kJ`),
  kWh: (J) => (asNumber(J) === null ? DASH : `${num(J / 3.6e6, 4)} kWh`),
  g: (kg) => (asNumber(kg) === null ? DASH : `${num(kg * 1000, 4)} g`),
  litres: (m3) => (asNumber(m3) === null ? DASH : `${num(m3 * 1000, 4)} l`),
  s: (s) => duration(s),
  pct,
  num,
  eur,
  si: formatSI,
};
