# Ultron on Plint: node relays between the page, the model that thinks and VoiceStudio.
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund
COPY server.js ./
COPY public ./public
ENV PORT=5230 DATA_DIR=/data NODE_ENV=production
USER node
EXPOSE 5230
CMD ["node", "server.js"]
