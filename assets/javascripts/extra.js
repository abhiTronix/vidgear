/* 
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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
    source: 'https://rawcdn.githack.com/abhiTronix/streamgear_chunks/503f9640b0be4350d4d6d04fecff3f9c4c4cd11c/files/dash_out.mpd',
    plugins: [DashShakaPlayback, LevelSelector],
    shakaConfiguration: {
        streaming: {
            rebufferingGoal: 10
        }
    },
    shakaOnBeforeLoad: function(shaka_player) {
        // shaka_player.getNetworkingEngine().registerRequestFilter() ...
    },
    width: '100%',
    height: 'auto',
    parentId: '#player'
});

var player_stab = new Clappr.Player({
    source: 'https://rawcdn.githack.com/abhiTronix/streamgear_chunks/5900a1c70d74980c9d50207aee0941b728b72a88/files2/dash_out.mpd',
    plugins: [DashShakaPlayback],
    shakaConfiguration: {
        streaming: {
            rebufferingGoal: 5
        }
    },
    shakaOnBeforeLoad: function(shaka_player) {
        // shaka_player.getNetworkingEngine().registerRequestFilter() ...
    },
    width: '100%',
    height: 'auto',
    parentId: '#player_stab'
});