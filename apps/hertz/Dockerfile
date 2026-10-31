# Hertz on Plint: the instrument runs in the browser; node only serves it and keeps songs.
FROM node:22-alpine
WORKDIR /app
COPY server.js ./
COPY public ./public
ENV PORT=5200 DATA_DIR=/data NODE_ENV=production
USER node
EXPOSE 5200
CMD ["node", "server.js"]
