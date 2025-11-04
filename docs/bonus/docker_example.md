<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
-->

# Containerizing VidGear with Docker

<figure>
  <img src="https://raw.githubusercontent.com/abhiTronix/vidgear-docker-example/main/assets/docker-logo.png" loading="lazy" alt="Docker Logo" class="center" style="width: 50%;"/>
</figure>

> This page serves as a template and inspiration for containerizing VidGear applications with Docker, demonstrating best practices for building production-ready video processing containers.

!!! info "Complete Example Repository"
    A full working implementation with detailed documentation is available at: [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example)

&nbsp;

## Overview

Docker containerization enables VidGear applications to run consistently across different environments with all required dependencies (FFmpeg, GStreamer, OpenCV) bundled together. This approach provides:

- **Reproducible Builds**: Consistent environment across development, testing, and production
- **Dependency Isolation**: All multimedia libraries packaged within the container
- **Easy Deployment**: Single container image that works anywhere Docker runs
- **Scalability**: Simple horizontal scaling for video processing workloads

&nbsp;

## Key Implementation Concepts

### Multi-Stage Dockerfile

Use multi-stage builds to minimize final image size while maintaining build flexibility:

```dockerfile
# Build stage - install build dependencies
FROM python:3.9-slim as builder
RUN apt-get update && apt-get install -y \
    build-essential ffmpeg libsm6 libxext6

# Production stage - copy only runtime dependencies
FROM python:3.9-slim
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

&nbsp;

### Environment-Based Configuration

Use environment variables for flexible configuration without rebuilding images:

```python
import os
from vidgear.gears import CamGear, WriteGear

# Configuration from environment
video_url = os.getenv('VIDEO_URL', 'default-video-url')
output_path = os.getenv('OUTPUT_FILE', '/app/output/output.mp4')
codec = os.getenv('OUTPUT_CODEC', 'libx264')

# Initialize VidGear with configuration
stream = CamGear(source=video_url, logging=True).start()
writer = WriteGear(output_filename=output_path, compression_mode=True, 
                   logging=True, **{"-vcodec": codec})
```

&nbsp;

### Docker Compose Orchestration

Simplify container management with Docker Compose:

```yaml
version: '3.8'
services:
  vidgear-app:
    build: .
    volumes:
      - ./output:/app/output
    env_file:
      - .env
    environment:
      - VIDEO_URL=${VIDEO_URL}
      - OUTPUT_CODEC=${OUTPUT_CODEC}
```

&nbsp;

## Quick Start

Basic steps to containerize your VidGear application:

```bash
# 1. Clone the example repository
git clone https://github.com/abhiTronix/vidgear-docker-example.git
cd vidgear-docker-example

# 2. Configure environment variables
cp .env.example .env

# 3. Build and run
docker-compose up
```

&nbsp;

## Best Practices

### Security

- Run containers as non-root user
- Use specific base image versions (avoid `latest` tags)
- Keep base images updated for security patches
- Minimize attack surface by removing unnecessary tools

### Performance

- Use volume mounts for output files to avoid container bloat
- Consider tmpfs mounts for temporary processing files
- Optimize FFmpeg settings for your use case
- Use appropriate resource limits (CPU, memory)

### Development Workflow

- Separate development and production Dockerfiles if needed
- Use `.dockerignore` to exclude unnecessary files
- Implement health checks for container monitoring
- Tag images properly for version tracking

&nbsp;

## Additional Resources

For complete implementation details including:

- Full Dockerfile with all dependencies
- Comprehensive environment configuration
- Testing setup with pytest
- CI/CD pipeline examples
- Troubleshooting guides
- Multiple usage examples

Visit the [**vidgear-docker-example repository**](https://github.com/abhiTronix/vidgear-docker-example) on GitHub.

&nbsp;
