# CNC Studio on Plint. Upstream app as-is; only this file and .dockerignore are Plint's.
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY server.js ./
COPY prompts ./prompts
COPY public ./public
ENV HOST=0.0.0.0 PORT=5178 NODE_ENV=production
USER node
EXPOSE 5178
CMD ["node", "server.js"]
