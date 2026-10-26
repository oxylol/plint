// Decoding for old CNC files: UTF-8, Windows-1252, or DOS code page 850
// (CncSimulator writes its tool registers and string tables in CP850).

const CP850_HIGH = [
  'ÇüéâäàåçêëèïîìÄÅ', 'ÉæÆôöòûùÿÖÜø£Ø×ƒ', 'áíóúñÑªº¿®¬½¼¡«»', '░▒▓│┤ÁÂÀ©╣║╗╝¢¥┐',
  '└┴┬├─┼ãÃ╚╔╩╦╠═╬¤', 'ðÐÊËÈıÍÎÏ┘┌█▄¦Ì▀', 'ÓßÔÒõÕµþÞÚÛÙýÝ¯´', '­±‗¾¶§÷¸°¨·¹³²■ ',
].join('');

export function decodeCp850(bytes) {
  let s = '';
  for (const b of bytes) s += b < 0x80 ? String.fromCharCode(b) : CP850_HIGH[b - 0x80];
  return s;
}

// Pick the most plausible decoding for a byte buffer.
export function decodeLegacyText(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  let high = 0, c1 = 0;
  for (const b of bytes) { if (b >= 0x80) { high++; if (b < 0xa0) c1++; } }
  if (!high) return new TextDecoder('utf-8').decode(bytes);
  const utf = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  if (!utf.includes('�')) return utf;
  // 0x80–0x9F are mostly unused in Windows-1252 text but are Swedish letters (ä å Ä Å ö …) in CP850
  return c1 / high > 0.3 ? decodeCp850(bytes) : new TextDecoder('windows-1252').decode(bytes);
}

export function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
