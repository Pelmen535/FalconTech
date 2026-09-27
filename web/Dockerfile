# Клиентский слой: только статика и обратный прокси, без Python и без весов.
# Тег зафиксирован по дайджесту — ссылаться на latest запрещено (ответ 39).
ARG WEB_BASE_IMAGE=nginx:1.27.4-alpine@sha256:4ff102c5d78d254a6f0da062b3cf39eaf07f01eec0927fd21e219d0af8bc0591
FROM ${WEB_BASE_IMAGE}

# Контекст сборки — корень репозитория: страница лежит рядом с сервисом, а не дублируется.
COPY service/static/ /usr/share/nginx/html/static/
COPY service/static/index.html /usr/share/nginx/html/index.html
COPY web/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
