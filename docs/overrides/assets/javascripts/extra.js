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

var player = new Clappr.Player({
  source: 'https://rawcdn.githack.com/abhiTronix/vidgear-docs-additionals/fbcf0377b171b777db5e0b3b939138df35a90676/streamgear_video_chunks/streamgear_dash.mpd',
  plugins: [DashShakaPlayback, LevelSelector],
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
  parentId: '#player',
  poster: 'https://rawcdn.githack.com/abhiTronix/vidgear-docs-additionals/674250e6c0387d0d0528406eec35bc580ceafee3/streamgear_video_chunks/hd_thumbnail.jpg',
  preload: 'metadata',
});

var player_stab = new Clappr.Player({
  source: 'https://rawcdn.githack.com/abhiTronix/vidgear-docs-additionals/fbcf0377b171b777db5e0b3b939138df35a90676/stabilizer_video_chunks/stabilizer_dash.mpd',
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
  poster: 'https://rawcdn.githack.com/abhiTronix/vidgear-docs-additionals/94bf767c28bf2fe61b9c327625af8e22745f9fdf/stabilizer_video_chunks/hd_thumbnail_2.png',
  preload: 'metadata',
});
