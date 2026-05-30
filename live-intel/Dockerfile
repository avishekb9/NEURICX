# NEURICX live-intel — Cloud Run container
# node:https-only server, zero npm deps. Cloud Run injects PORT (8080); we bind
# 0.0.0.0 via env (the server defaults to 127.0.0.1 for local safety).
FROM node:22-alpine
WORKDIR /app
COPY server/ ./server/
COPY web/ ./web/
ENV HOST=0.0.0.0
ENV PORT=8080
# GOOGLE_API_KEY is provided at deploy time via --set-env-vars / Secret Manager,
# never baked into the image.
EXPOSE 8080
CMD ["node", "server/neuricx-server.mjs"]
