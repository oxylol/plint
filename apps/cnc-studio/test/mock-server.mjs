// Starts the server with a scripted Claude (no API key needed): node test/mock-server.mjs
process.env.MOCK_CLAUDE = '1';
process.env.PORT ||= '5179';
await import('../server.js');
