var pc = null;
var dataChannel = null;

function negotiate() {
    pc.addTransceiver('video', {direction: 'sendonly'});
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
    }

    pc = new RTCPeerConnection(config);

    // Create a data channel for receiving data from the server
    dataChannel = pc.createDataChannel('data');
    dataChannel.onmessage = function(event) {
        let data = JSON.parse(event.data);
        if (data.id) {
            let labelElement = document.createElement('div');
            labelElement.textContent = `ID: ${data.id}, X: ${data.label_x}, Y: ${data.label_y}`;
            labelElement.style.position = 'absolute';
            labelElement.style.left = `${data.label_x}px`;
            labelElement.style.top = `${data.label_y}px`;
            labelElement.style.color = 'red';
            labelElement.style.background = 'rgba(255, 255, 255, 0.7)';
            labelElement.style.padding = '2px 5px';
            labelElement.style.borderRadius = '5px';
            document.body.appendChild(labelElement);
        }
    };

    var constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
            facingMode: "user"  // Use the rear camera
        }
    };

    // Add local video track to the connection
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        stream.getTracks().forEach(track => pc.addTrack(track, stream));
        document.getElementById('start').style.display = 'none';
        return negotiate();
    }).then(function() {
        document.getElementById('stop').style.display = 'inline-block';
    }).catch(function(err) {
        alert('Error in starting the connection: ' + err.message);
    });
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    setTimeout(function() {
        pc.close();
    }, 500);
}
