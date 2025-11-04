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

# VidGear Docker Streamer

<figure>
  <img src="https://raw.githubusercontent.com/abhiTronix/vidgear-docker-example/main/assets/docker-logo.png" loading="lazy" alt="Docker Logo" class="center" style="width: 50%;"/>
</figure>

A production-ready Docker application template for the VidGear framework that demonstrates video streaming, processing, and encoding with audio support. This template showcases best practices for containerizing VidGear applications with FFmpeg, GStreamer, and OpenCV.

!!! info "Repository"
    The complete source code and examples are available at: [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example)

&nbsp;

## Features

- :fontawesome-solid-film: **Video Streaming**: Stream videos from YouTube, Twitch, and other platforms using yt-dlp
- :fontawesome-solid-music: **Audio Support**: Automatic audio extraction and merging
- :fontawesome-solid-wrench: **Flexible Configuration**: Environment-based configuration for easy customization
- :fontawesome-brands-docker: **Docker Compose**: Simple orchestration with docker-compose
- :fontawesome-solid-box: **Multi-stage Build**: Optimized Docker image with minimal size
- :fontawesome-solid-lock: **Security**: Non-root user execution and minimal attack surface
- :fontawesome-solid-flask: **Testing**: Comprehensive test suite with pytest
- :fontawesome-solid-robot: **CI/CD**: GitHub Actions for automated testing and releases
- :fontawesome-solid-book: **Documentation**: Detailed configuration and usage guides

&nbsp;

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** 20.10 or higher
- **Docker Compose** 2.0 or higher (optional but recommended)
- **Git** (for cloning the repository)

&nbsp;

## Quick Start

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/abhiTronix/vidgear-docker-example.git
cd vidgear-docker-example
```

**Step 2: Configure environment**

```bash
cp .env.example .env
# Edit .env with your preferred settings
nano .env  # or use your favorite editor
```

**Step 3: Create output directory**

```bash
mkdir -p output
```

&nbsp;

### Usage Options

#### Option 1: Using Docker Compose (Recommended)

=== "Build and Run"

    ```bash
    # Build and run
    docker-compose up

    # Run in detached mode
    docker-compose up -d
    ```

=== "View Logs"

    ```bash
    # View logs
    docker-compose logs -f
    ```

=== "Stop Container"

    ```bash
    # Stop the container
    docker-compose down
    ```

&nbsp;

#### Option 2: Using Docker CLI

```bash
# Build the image
docker build -t vidgear-streamer .

# Run the container
docker run -v "$(pwd)/output:/app/output" --env-file .env vidgear-streamer

# Run with specific video URL
docker run -v "$(pwd)/output:/app/output" \
  -e VIDEO_URL="https://youtu.be/your-video-id" \
  vidgear-streamer
```

&nbsp;

#### Option 3: Using Makefile

```bash
# Build the image
make build

# Run the container
make run

# View logs
make logs

# Clean up
make clean

# Run tests
make test
```

&nbsp;

## Configuration

All configuration is done through environment variables. The following table lists the key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_URL` | Source video URL | `https://youtu.be/xvFZjo5PgG0` |
| `OUTPUT_FILE` | Output file path | `/app/output/vidgear_output.mp4` |
| `VIDEO_STREAM_QUALITY` | Video quality (best/720p/1080p) | `best` |
| `AUDIO_STREAM_QUALITY` | Audio quality | `bestaudio` |
| `OUTPUT_CODEC` | Video codec (libx264/libx265) | `libx264` |
| `AUDIO_CODEC` | Audio codec (aac/mp3) | `aac` |
| `FRAME_LIMIT` | Max frames to process (0=unlimited) | `0` |
| `VERBOSE` | Enable verbose logging | `false` |

!!! tip "For detailed configuration options, see the [CONFIGURATION.md](https://github.com/abhiTronix/vidgear-docker-example/blob/main/docs/CONFIGURATION.md) in the repository."

&nbsp;

## Usage Examples

### Example 1: Process YouTube Video

```bash
docker run -v "$(pwd)/output:/app/output" \
  -e VIDEO_URL="https://youtu.be/dQw4w9WgXcQ" \
  -e VIDEO_STREAM_QUALITY="720p" \
  -e FRAME_LIMIT="300" \
  vidgear-streamer
```

&nbsp;

### Example 2: High-Quality Processing

```bash
docker run -v "$(pwd)/output:/app/output" \
  -e VIDEO_URL="https://youtu.be/your-video" \
  -e VIDEO_STREAM_QUALITY="1080p" \
  -e OUTPUT_CODEC="libx265" \
  -e AUDIO_CODEC="aac" \
  -e VERBOSE="true" \
  vidgear-streamer
```

&nbsp;

### Example 3: Quick Test Processing

```bash
docker run -v "$(pwd)/output:/app/output" \
  -e VIDEO_URL="https://youtu.be/test-video" \
  -e FRAME_LIMIT="100" \
  -e VERBOSE="true" \
  vidgear-streamer
```

!!! example "More Examples"
    For additional usage examples, check the [examples/](https://github.com/abhiTronix/vidgear-docker-example/tree/main/examples) directory in the repository.

&nbsp;

## Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_streamer.py::test_streamer_initialization
```

&nbsp;

### Run Tests in Docker

```bash
# Using docker-compose
docker-compose -f docker-compose.test.yml up

# Using Makefile
make test
```

&nbsp;

## Development

### Building from Source

```bash
# Build with default settings
docker build -t vidgear-streamer .

# Build with custom build args
docker build \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t vidgear-streamer:dev .
```

&nbsp;

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# Run the application locally
python -m app.streamer

# Run tests
pytest
```

&nbsp;

## Troubleshooting

### Common Issues

!!! warning "Permission Denied Errors"
    
    **Solution**: Fix permissions on output directory
    
    ```bash
    chmod -R 755 output/
    ```

!!! warning "Container Exits Immediately"
    
    **Solution**: Check logs and run with verbose mode
    
    ```bash
    # Check logs
    docker-compose logs

    # Run with verbose mode
    docker-compose up --env VERBOSE=true
    ```

!!! warning "Video URL Not Supported"
    
    **Solution**: Test URL with yt-dlp first
    
    ```bash
    # Test URL with yt-dlp first
    yt-dlp -F "your-video-url"
    ```

&nbsp;

## Performance

Typical performance metrics you can expect:

| Metric | Value |
|--------|-------|
| **CPU Usage** | 150-400% (multi-core) |
| **Memory** | 500MB - 2GB (depends on video quality) |
| **Processing Speed** | Real-time to 2x realtime (depends on codec) |
| **Disk I/O** | Moderate (depends on output format) |

&nbsp;

## Contributing

We welcome contributions to the VidGear Docker Streamer project! Here's how you can help:

- :fontawesome-solid-bug: Report bugs
- :fontawesome-solid-lightbulb: Suggest features
- :fontawesome-solid-code-pull-request: Submit pull requests
- :fontawesome-solid-book: Improve documentation

!!! info "For detailed contribution guidelines, see [CONTRIBUTING.md](https://github.com/abhiTronix/vidgear-docker-example/blob/main/CONTRIBUTING.md) in the repository."

&nbsp;

## Roadmap

Future enhancements planned for the VidGear Docker Streamer:

- [ ] Add GPU acceleration support (NVIDIA/AMD)
- [ ] Implement real-time streaming (RTMP/HLS)
- [ ] Add web UI for configuration
- [ ] Support for multiple video sources
- [ ] Add video filters and effects
- [ ] Implement batch processing
- [ ] Add monitoring and metrics (Prometheus)
- [ ] Kubernetes deployment templates

&nbsp;

## Acknowledgments

This project builds upon several excellent open-source projects:

- [**VidGear**](https://github.com/abhiTronix/vidgear) - High-performance video processing framework
- [**FFmpeg**](https://ffmpeg.org/) - Multimedia processing
- [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) - Video downloading
- [**OpenCV**](https://opencv.org/) - Computer vision library

&nbsp;

## Support

Need help or have questions?

- :fontawesome-solid-book: [**Documentation**](https://github.com/abhiTronix/vidgear-docker-example/tree/main/docs)
- :fontawesome-solid-bug: [**Issue Tracker**](https://github.com/abhiTronix/vidgear-docker-example/issues)
- :fontawesome-solid-comments: [**Discussions**](https://github.com/abhiTronix/vidgear-docker-example/discussions)

&nbsp;

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/abhiTronix/vidgear-docker-example/blob/main/LICENSE) file for details.

&nbsp;

---

Made with :heart: by the VidGear community
