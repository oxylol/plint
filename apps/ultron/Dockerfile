# Ultron on Plint: node relays between the page, celeris-1 and VoiceStudio. No dependencies.
FROM node:22-alpine
WORKDIR /app
COPY server.js ./
COPY public ./public
ENV PORT=5230 DATA_DIR=/data NODE_ENV=production
USER node
EXPOSE 5230
CMD ["node", "server.js"]
