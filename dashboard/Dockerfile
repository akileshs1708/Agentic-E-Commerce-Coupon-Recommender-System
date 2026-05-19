FROM nginx:alpine

# Copy nginx config
COPY dashboard/nginx.conf /etc/nginx/nginx.conf

# Copy dashboard HTML
COPY dashboard/index.html /usr/share/nginx/html/index.html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
