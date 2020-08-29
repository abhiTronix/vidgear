var player = new Clappr.Player({
    source: 'https://rawcdn.githack.com/abhiTronix/streamgear_chunks/503f9640b0be4350d4d6d04fecff3f9c4c4cd11c/files/dash_out.mpd',
    plugins: [DashShakaPlayback, LevelSelector],
    shakaConfiguration: {
        streaming: {
            rebufferingGoal: 15
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
            rebufferingGoal: 15
        }
    },
    shakaOnBeforeLoad: function(shaka_player) {
        // shaka_player.getNetworkingEngine().registerRequestFilter() ...
    },
    width: '100%',
    height: 'auto',
    parentId: '#player_stab'
});