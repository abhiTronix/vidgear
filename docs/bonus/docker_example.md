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

# Containerizing VidGear with Docker :material-docker:

<div align="center">
<img src="https://abhitronix.github.io/vidgear/latest/assets/images/vidgear.png" loading="lazy" alt="Vidgear Logo" style="width: 85%;"/>
</div> <div align="right">
<img src="https://raw.githubusercontent.com/abhiTronix/vidgear-docker-example/refs/heads/main/docs/assets/docker-logo-blue.png" loading="lazy" alt="Docker Logo" style="width: 20%;"/>
</div>

> This guide addresses the unique challenges of containerizing VidGear applications, particularly around building OpenCV with proper GStreamer and FFmpeg support. Learn the patterns and pitfalls before you start.

!!! example "Complete Reference Implementation :fontawesome-solid-book-bookmark:"

    **If you're looking for a complete implementation with:**

    - [x] **Full Dockerfile**: Multi-stage build with OpenCV, FFmpeg, GStreamer
    - [x] **Comprehensive Testing**: pytest suite with mocks and fixtures
    - [x] **CI/CD Pipeline**: GitHub Actions for automated builds
    - [x] **Multiple Examples**: Different streaming scenarios
    - [x] **Troubleshooting Guide**: Common issues and solutions
    - [x] **Performance Tuning**: Optimization strategies

    **The reference implementation is available at:** [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example)

&nbsp;

## VidGear - The Dockerization Challenge :material-trending-up:

The biggest challenge in containerizing VidGear isn't VidGear itself—it's **getting OpenCV with proper video backend support**. Most pip-installed OpenCV builds lack GStreamer support, which limits VidGear's capabilities.

### Why This Matters?

VidGear relies on OpenCV's video I/O backends:

- **FFmpeg backend**: Essential for reading/writing most video formats
- **GStreamer backend**: Required for network streams, pipelines, and advanced processing
- **Codec support**: Proper codec libraries (libx264, libx265, etc.) must be available

Simply doing `pip install opencv-python` in a container gives you a minimal build that **will not work properly** with many VidGear features.


&nbsp;

## Three Approaches to Building :simple-docker: Docker Containers

### Approach 1: Use Pre-built OpenCV with Video Support (Simplest)

Use the pre-built OpenCV packages from VidGear's CI releases that include GStreamer and FFmpeg support:

!!! example "Complete Reference Implementation :material-book-open-page-variant:"
    A complete working example demonstrating these patterns is available at: [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example)

???+ info "Pre-built OpenCV packages are built with full GStreamer and FFmpeg support"
    The pre-built OpenCV packages are available for various supported Python versions here: [**VidGear OpenCV CI Releases**](https://github.com/abhiTronix/OpenCV-CI-Releases/releases)

    This approach simplifies the Dockerfile significantly since you don't need to build OpenCV from source. Moreover, we use the same pre-built OpenCV binaries in Vidgear's CI tests ensuring maximum compatibility.

    !!! warning "Please note that these are non-commercial builds and are intended for development and testing purposes only. Always verify compliance with licensing terms for production use."

    !!! failure "Pre-built OpenCV packages may not be available work on all linux distributions or architectures. In such cases, you may need to build OpenCV from source as described in Approach 2."

??? tip "Quick Start Docker commands for Reference Implementation"

    Here are some quick commands to build and run the Docker container using the provided Dockerfile and docker-compose setup:

    ```bash
    # Build the image
    docker build -t my-vidgear-app .

    # Run with basic configuration
    docker run -v "$(pwd)/output:/app/output" \
      -e VIDEO_URL="https://youtu.be/dQw4w9WgXcQ" \
      my-vidgear-app

    # Run with docker-compose
    docker-compose up

    # Run in background and view logs
    docker-compose up -d
    docker-compose logs -f

    # Stop and clean up
    docker-compose down

    # Rebuild after changes
    docker-compose up --build
    ```


??? danger "Essential Dependencies Breakdown for Pre-built OpenCV Packages"

    Ensure all these dependencies are installed in the builder stage when building OpenCV from source. Missing any of these can lead to runtime errors:

    1. **FFmpeg Core:**
      ```dockerfile
      RUN apt-get install -y \
          ffmpeg \              # FFmpeg binary
          libavcodec-dev \        # Codec library
          libavformat-dev \     # Container format library
          libavutil-dev \       # Utility functions
          libswscale-dev \      # Scaling/color conversion
          libswresample-dev     # Audio resampling
      ```
    2. **Video Codecs:**
      ```dockerfile
      RUN apt-get install -y \
          libx264-dev \         # H.264 encoder
          libx265-dev \         # H.265/HEVC encoder (if available)
          libxvidcore-dev \     # Xvid encoder
          libvpx-dev \          # VP8/VP9 encoder
          libtheora-dev         # Theora encoder
      ```
    3. **GStreamer (For Advanced Features)**
      ```dockerfile
      RUN apt-get install -y \
          libgstreamer1.0-dev \                    # GStreamer core
          libgstreamer-plugins-base1.0-dev \       # Base plugins development
          gstreamer1.0-plugins-base \              # Base plugins runtime
          gstreamer1.0-plugins-good \              # Good plugins
          gstreamer1.0-plugins-bad \               # Bad plugins (useful codecs)
          gstreamer1.0-plugins-ugly \              # Ugly plugins (patent issues)
          gstreamer1.0-libav \                     # FFmpeg wrapper
          gstreamer1.0-tools                       # CLI tools for debugging
      ```
    4. **Image Format Support**
      ```dockerfile
      RUN apt-get install -y \
          libjpeg-dev \         # JPEG
          libpng-dev \          # PNG
          libtiff-dev \         # TIFF
          libwebp-dev           # WebP
      ```
    5. **Math Libraries (OpenCV Dependencies)**
      ```dockerfile
      RUN apt-get install -y \
          libatlas-base-dev \   # Linear algebra
          libopenblas-dev       # Optimized BLAS
      ```

```dockerfile
# Stage 1: Build OpenCV with full video support 
# Supported Linux Distributions: Ubuntu 22.04 (4)
FROM ubuntu:22.04 AS opencv-installer

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies first
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl \
    # FFmpeg and codecs
    ffmpeg libavcodec58 libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    libx264-dev libxvidcore-dev \
    # GStreamer
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    # Image libraries
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    # Math libraries
    libatlas-base-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install pre-built OpenCV with GStreamer support (1)
RUN PYTHONSUFFIX=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))') && \
    echo "Detecting Python $PYTHONSUFFIX" && \
    LATEST_VERSION=$(curl -sL https://api.github.com/repos/abhiTronix/OpenCV-CI-Releases/releases/latest | \
        grep "OpenCV-.*-$PYTHONSUFFIX.*.deb" | \
        grep -Eo "(http|https)://[a-zA-Z0-9./?=_%:-]*" | head -1) && \
    echo "Downloading: $LATEST_VERSION" && \
    curl -LO $LATEST_VERSION && \
    OPENCV_FILENAME=$(basename "$LATEST_VERSION") && \
    python3 -m pip install numpy && \
    dpkg -i "$OPENCV_FILENAME" && \
    # Link OpenCV to Python dist-packages
    ln -sf /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so /usr/lib/python3/dist-packages/ && \
    ldconfig && \ 
    # Verify installation
    python3 -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"

# Stage 2: Runtime image
# Supported Linux Distributions: Ubuntu 22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install ONLY runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libatlas-base-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenCV from builder
COPY --from=opencv-installer /usr/local/lib/ /usr/local/lib/
COPY --from=opencv-installer /usr/local/include/ /usr/local/include/
COPY --from=opencv-installer /usr/lib/python3/dist-packages/ /usr/lib/python3/dist-packages/

# Update library cache
RUN ldconfig

WORKDIR /app

# Install VidGear and dependencies (2)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify everything works (3)
RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, '| GStreamer:', cv2.videoio_registry.getBackendName(cv2.CAP_GSTREAMER))" && \
    python3 -c "import vidgear; print('VidGear:', vidgear.__version__)" && \
    ffmpeg -version | head -n 1

# Security: non-root user (5)
RUN useradd -r -m -u 1000 vidgear && \
    mkdir -p /app/output && \
    chown -R vidgear:vidgear /app

COPY app/ ./app/
RUN chown -R vidgear:vidgear /app

USER vidgear

CMD ["python3", "-m", "app.streamer"]
```

1. **⚠️ Proper Python Library Linking:** OpenCV must be accessible to Python. The linking step is crucial:

    ```dockerfile
    # Detect Python version dynamically
    RUN PYTHONSUFFIX=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))') && \
        # Create symlinks from OpenCV install location to Python's site-packages
        ln -sf /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so /usr/lib/python3/dist-packages/ && \
        # Update dynamic linker cache
        ldconfig
    ```

2. **⚠️ Dependency Installation Order:** Install dependencies in the correct order to avoid conflicts with shared libraries:

    ```dockerfile
    # 1. System dependencies first
    RUN apt-get update && apt-get install -y ffmpeg gstreamer1.0-tools ...

    # 2. Python build dependencies (if building packages)
    RUN pip install --upgrade pip setuptools wheel

    # 4. VidGear and its dependencies
    RUN pip install vidgear[asyncio]

    # 5. Optional: yt-dlp for streaming from YouTube/Twitch
    RUN pip install --upgrade "yt-dlp[default]"
    ```

3. **⚠️ Verification Steps:** Always verify installations work before copying your application:

    ```dockerfile
    RUN python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
        python3 -c "import cv2; print(f'GStreamer available: {cv2.videoio_registry.getBackendName(cv2.CAP_GSTREAMER)}')" && \
        python3 -c "import vidgear; print(f'VidGear: {vidgear.__version__}')" && \
        ffmpeg -version | head -n 1
    ```

    If any of these fail, your container won't work properly with VidGear.

4. **⚠️ Supported Linux Distributions:** The current pre-built OpenCV packages are built against Ubuntu 22.04 (Jammy Jellyfish). If you're using a different distribution or version, you may need to build OpenCV from source as described in [Approach 2](#approach-2-build-your-own-opencv-from-source-most-flexible).

5. **🗝️ Security:** Running as non-root user to minimize security risks.

!!! info "Why Multi-Stage Build?"
    - **Builder stage**: Includes build tools and downloads OpenCV with full support
    - **Runtime stage**: Contains only runtime libraries, reducing final image size by ~500MB
    - OpenCV binaries are copied between stages, not rebuilt

&thinsp;

### Approach 2: Build your Own OpenCV from Source (Most Flexible)

If pre-built OpenCV packages are not available for your distribution or architecture, you can build OpenCV from source with the required video backend support.

Checkout this detailed script that automates building OpenCV with GStreamer and FFmpeg support: [**`create_opencv.sh`**](https://github.com/abhiTronix/OpenCV-CI-Releases/blob/main/create_opencv.sh)

!!! tip "You could integrate this script into a multi-stage Dockerfile similar to [Approach 1](#approach-1-use-pre-built-opencv-with-video-support-simplest), ensuring all dependencies are installed in the builder stage before running the script."

&thinsp;

### Approach 3: Minimal Build (Limited Functionality)

If you only need basic video file I/O and don't require GStreamer:

???+ failure "Limitations of Minimal Build"
    This approach uses pip's opencv-python which lacks GStreamer support. You **cannot** access:

    - [ ] Network streams via GStreamer pipelines
    - [ ] Youtube/Twitch live streams
    - [ ] Advanced camera controls
    - [ ] Some specialized VidGear features

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ... rest of your app
```

&nbsp;

## Application Configuration Pattern (:fontawesome-solid-square-root-variable: Environment-Based)

Structure your VidGear application to read all configuration from environment variables:

???+ "Environment File Template (Basics)"

    The following is a sample `.env.example` file from [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example) illustrating common configuration options for a VidGear streaming application:

    ```bash
    # ==========================================
    # Input Source Configuration
    # ==========================================
    VIDEO_URL=https://youtu.be/xvFZjo5PgG0
    VIDEO_STREAM_QUALITY=best        # Options: best, 1080p, 720p, 480p
    AUDIO_STREAM_QUALITY=bestaudio   # Options: bestaudio, 192, 128

    # ==========================================
    # Output Configuration
    # ==========================================
    OUTPUT_FILE=/app/output/vidgear_output.mp4
    OUTPUT_CODEC=libx264             # Options: libx264, libx265, libvpx
    AUDIO_CODEC=aac                  # Options: aac, mp3, libopus

    # ==========================================
    # Processing Options
    # ==========================================
    FRAME_LIMIT=0                    # 0 = process entire video
    VERBOSE=false                    # Enable debug logging
    ```
    
    It depicts how to configure input source, output parameters, and processing options using environment variables. This pattern allows easy customization without modifying the code.

!!! example "Complete Reference Implementation :material-book-open-page-variant:"
    A complete working example demonstrating these patterns is available at: [**abhiTronix/vidgear-docker-example**](https://github.com/abhiTronix/vidgear-docker-example)

```python
# app/streamer.py
import os
from pathlib import Path
from yt_dlp import YoutubeDL
from vidgear.gears import CamGear, WriteGear

class VideoStreamer:
  def __init__(self):
    # Input configuration
    self.video_url = os.getenv('VIDEO_URL')
    self.video_quality = os.getenv('VIDEO_STREAM_QUALITY', 'best')
    self.audio_quality = os.getenv('AUDIO_STREAM_QUALITY', 'bestaudio')
    
    # Output configuration
    self.output_file = Path(os.getenv('OUTPUT_FILE', '/app/output/output.mp4'))
    self.output_video = self.output_file.with_suffix('.tmp.mp4')
    self.output_audio = self.output_file.with_suffix('.tmp.aac')
    self.output_codec = os.getenv('OUTPUT_CODEC', 'libx264')
    self.audio_codec = os.getenv('AUDIO_CODEC', 'aac')
    
    # Processing options
    self.frame_limit = int(os.getenv('FRAME_LIMIT', '0'))
    self.verbose = os.getenv('VERBOSE', 'false').lower() == 'true'
    self.framerate = 30  # Default framerate
    
    # Ensure output directory exists
    self.output_file.parent.mkdir(parents=True, exist_ok=True)
  
  def download_audio(self):
    """Download audio stream using yt-dlp"""
    ydl_opts = {
      "format": self.audio_quality,
      "quiet": not self.verbose,
      "no_warnings": True,
      "outtmpl": str(self.output_audio),
    }
    with YoutubeDL(ydl_opts) as ydl:
      ydl.download([self.video_url])
  
  def start_stream(self):
    """Initialize CamGear with options"""
    stream_options = {
      "STREAM_RESOLUTION": self.video_quality,
    }
    
    self.stream = CamGear(
      source=self.video_url,
      stream_mode=True,
      logging=self.verbose,
      **stream_options
    ).start()
    
    # Get video metadata
    video_metadata = self.stream.ytv_metadata
    self.framerate = video_metadata.get("fps", 30)
  
  def start_writer(self):
    """Initialize WriteGear with output parameters"""
    output_params = {
      "-input_framerate": self.framerate,
      "-c:v": self.output_codec,
    }
    
    self.writer = WriteGear(
      output=str(self.output_video),
      compression_mode=True,
      logging=self.verbose,
      **output_params
    )
  
  def combine_audio_video(self):
    """Combine audio and video into final output"""
    ffmpeg_command = [
      "-y",
      "-i", str(self.output_video),
      "-i", str(self.output_audio),
      "-c:v", "copy",
      "-c:a", "copy",
      "-map", "0:v:0",
      "-map", "1:a:0",
      "-shortest",
      str(self.output_file),
    ]
    self.writer.execute_ffmpeg_cmd(ffmpeg_command)
  
  def process(self):
    """Main processing loop"""
    self.download_audio()
    self.start_stream()
    self.start_writer()
    
    frame_count = 0
    
    try:
      while True:
        frame = self.stream.read()
        
        if frame is None:
          break
        
        self.writer.write(frame)
        
        frame_count += 1
        if self.frame_limit > 0 and frame_count >= self.frame_limit:
          break
          
    finally:
      self.cleanup()
  
  def cleanup(self):
    """Clean up resources"""
    if hasattr(self, 'stream'):
      self.stream.stop()
    if hasattr(self, 'writer'):
      self.writer.close()
    
    # Combine audio and video
    if self.output_audio.exists() and self.output_video.exists():
      self.combine_audio_video()
    
    # Remove temporary files
    if self.output_audio.exists():
      self.output_audio.unlink()
    if self.output_video.exists():
      self.output_video.unlink()

if __name__ == "__main__":
  streamer = VideoStreamer()
  streamer.process()
```

&nbsp;

## Common Pitfalls and Solutions

### Issue 1: "cv2 module not found" After Building

- [ ] **Cause**: OpenCV shared libraries not linked to Python's site-packages or missing

- [x] **Solution**: Verify linking and ldconfig was run:

```dockerfile
RUN PYTHONSUFFIX=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))') && \
    ln -sf /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so /usr/lib/python3/dist-packages/ && \
    ldconfig && \
    python3 -c "import cv2"  # This should not fail
```

### Issue 2: "GStreamer backend not available"

- [ ] **Cause**: OpenCV built without GStreamer support or GStreamer plugins missing

- [x] **Solution**: Use the pre-built OpenCV from VidGear CI releases ([Approach 1](#approach-1-use-pre-built-opencv-with-video-support-simplest)) or ensure GStreamer dev packages are installed during OpenCV compilation ([Approach 2](#approach-2-build-your-own-opencv-from-source-most-flexible)).

### Issue 3: "Permission denied" Writing Output Files

- [ ] **Cause**: Container running as non-root user but mounted volume has incorrect permissions

- [x] **Solution**: 

```bash
# On host
mkdir -p output
chmod 755 output

# In Dockerfile, match the UID
RUN useradd -m -u 1000 vidgear
USER vidgear
```

### Issue 4: "codec not found" Errors

- [ ] **Cause**: Missing codec libraries in runtime stage

- [x] **Solution**: Ensure all codec libraries from builder are available in runtime:
```dockerfile
# Runtime stage must include codec libraries
RUN apt-get install -y \
    libx264-dev \
    libavcodec-dev \
    ffmpeg
```

### Issue 5: Container Size Too Large (>2GB)

- [ ] **Cause**: Including build tools in final image

- [x] **Solution**: Use multi-stage build and only copy necessary binaries:
```dockerfile
# Don't include in runtime:
# build-essential, cmake, git, *-dev packages (except runtime libs)
```

&nbsp;

## Debugging Your Container

You could exec into the running container to test GStreamer, FFmpeg, and VidGear functionality directly.

### Test GStreamer Support

The following commands check if GStreamer is properly installed and if OpenCV can access the GStreamer backend:

```bash
# Enter running container
docker exec -it vidgear-streamer bash

# Check GStreamer plugins
gst-inspect-1.0 | grep -i video

# Test GStreamer pipeline
gst-launch-1.0 videotestsrc ! autovideosink

# Check OpenCV GStreamer backend
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

### Test FFmpeg Codecs

The following commands check if FFmpeg has the necessary encoders installed:

```bash
# Check available encoders
ffmpeg -encoders | grep -i h264

# Test encoding
ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 \
  -c:v libx264 -preset medium test.mp4
```

### Test VidGear Functionality

The following command tests if VidGear can read from a YouTube URL:

```bash
python3 -c "from vidgear.gears import CamGear; stream = CamGear(source="https://youtu.be/xvFZjo5PgG0").start(); frame = stream.read(); print(frame.shape); stream.stop()"
```

&nbsp;
