// STEP / IGES / BREP reader (OpenCascade compiled to WebAssembly) off the main thread.
importScripts('/vendor/occt/occt-import-js.js');
let occt = null;
onmessage = async (e) => {
  const { id, format, buffer, params } = e.data;
  try {
    occt ||= await occtimportjs({ locateFile: (f) => `/vendor/occt/${f}` });
    const bytes = new Uint8Array(buffer);
    const fn = { step: 'ReadStepFile', iges: 'ReadIgesFile', brep: 'ReadBrepFile' }[format];
    const r = occt[fn](bytes, format === 'brep' ? null : params);
    // only what the main thread needs (typed arrays transfer cheaply)
    const meshes = (r.meshes || []).map((m) => ({ attributes: { position: { array: Float32Array.from(m.attributes.position.array) } }, index: { array: Uint32Array.from(m.index.array) }, name: m.name }));
    postMessage({ id, result: { success: r.success, meshes } });
  } catch (err) {
    postMessage({ id, error: String(err?.message || err) });
  }
};
