# Accrete on Plint: the game runs in the browser and keeps itself there; node only hands out the files.
FROM node:22-alpine
WORKDIR /app
COPY server.js ./
COPY public ./public
ENV PORT=5240 NODE_ENV=production
USER node
EXPOSE 5240
CMD ["node", "server.js"]
