# Custom Nginx with automatic SSL certificate generation
FROM nginx:1.25-alpine

# Install openssl for certificate generation
RUN apk add --no-cache openssl

# Remove default nginx configuration
RUN rm -f /etc/nginx/conf.d/default.conf

# Copy nginx configuration
COPY conf.d /etc/nginx/conf.d

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Use custom entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

