# Multi-stage Dockerfile for Web UI (Next.js)

# Base stage
FROM node:18-alpine as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl

# Copy package files
COPY package*.json ./

# Development stage
FROM base as development

    # Install all dependencies (including dev dependencies)
    RUN npm install

# Copy source code
COPY . .

# Set environment variables
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start the development server
CMD ["npm", "run", "dev"]

# Production build stage
FROM base as build

# Install dependencies
RUN npm install --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine as production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl

# Copy built application from build stage
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package*.json ./
COPY --from=build /app/node_modules ./node_modules

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["npm", "start"]
