/* 
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
*/

// DASH StreamGear demo
var player_dash = new Clappr.Player({
  source: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/streamgear_video_segments/DASH/streamgear_dash.mpd',
  plugins: [DashShakaPlayback, LevelSelector],
  shakaConfiguration: {
    streaming: {
      rebufferingGoal: 30
    }
  },
  shakaOnBeforeLoad: function(shaka_player) {
    // shaka_player.getNetworkingEngine().registerRequestFilter() ...
  },
  levelSelectorConfig: {
    title: 'Quality',
    labels: {
      2: 'High', // 500kbps
      1: 'Med', // 240kbps
      0: 'Low', // 120kbps
    },
    labelCallback: function(playbackLevel, customLabel) {
      return customLabel; // High 720p
    }
  },
  width: '100%',
  height: '100%',
  parentId: '#player_dash',
  poster: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/streamgear_video_segments/DASH/hd_thumbnail.jpg',
  preload: 'metadata',
});

// HLS StremGear demo
var player_hls = new Clappr.Player({
  source: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/streamgear_video_segments/HLS/streamgear_hls.m3u8',
  plugins: [HlsjsPlayback, LevelSelector],
  hlsUseNextLevel: false,
  hlsMinimumDvrSize: 60,
  hlsRecoverAttempts: 16,
  hlsPlayback: {
    preload: true,
    customListeners: [],
  },
  playback: {
    extrapolatedWindowNumSegments: 2,
    triggerFatalErrorOnResourceDenied: false,
    hlsjsConfig: {
      // hls.js specific options
    },
  },
  levelSelectorConfig: {
    title: 'Quality',
    labels: {
      2: 'High', // 500kbps
      1: 'Med', // 240kbps
      0: 'Low', // 120kbps
    },
    labelCallback: function(playbackLevel, customLabel) {
      return customLabel; // High 720p
    }
  },
  width: '100%',
  height: '100%',
  parentId: '#player_hls',
  poster: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/streamgear_video_segments/HLS/hd_thumbnail.jpg',
  preload: 'metadata',
});

// DASH Stabilizer demo
var player_stab = new Clappr.Player({
  source: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/stabilizer_video_chunks/stabilizer_dash.mpd',
  plugins: [DashShakaPlayback],
  shakaConfiguration: {
    streaming: {
      rebufferingGoal: 30
    }
  },
  shakaOnBeforeLoad: function(shaka_player) {
    // shaka_player.getNetworkingEngine().registerRequestFilter() ...
  },
  width: '100%',
  height: '100%',
  parentId: '#player_stab',
  poster: 'https://bbcdn.githack.com/abhi_uno/vidgear-docs-additionals/raw/abc0c193ab26e21f97fa30c9267de6beb8a72295/stabilizer_video_chunks/hd_thumbnail.png',
  preload: 'metadata',
});