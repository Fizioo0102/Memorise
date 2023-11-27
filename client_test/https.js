import React, { useState, useEffect, useRef, FunctionComponent } from "react";
import { View, Button, Alert, StyleSheet } from "react-native";
import {
  MediaStream,
  RTCPeerConnection,
  RTCSessionDescription,
  RTCView,
  mediaDevices,
} from "react-native-webrtc";
import RNFetchBlob from "rn-fetch-blob";

const App: FunctionComponent = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [dataLabel, setDataLabel] = useState<string | null>(null);  // 추가된 상태 변수
  const pc = useRef<RTCPeerConnection | null>(null);

  const negotiate = async (): Promise<void> => {
    try {
      pc.current?.addTransceiver("video", { direction: "sendonly" }); //await 삭제함
      const offerOptions = {
        offerToReceiveAudio: true,
        offerToReceiveVideo: true,
      };

      const offer = await pc.current?.createOffer(offerOptions);
      if (!offer) throw new Error("Unable to create offer");

      await pc.current?.setLocalDescription(offer);

      const response = await RNFetchBlob.config({
        trusty: true,
      }).fetch(
        "POST",
        "https://70.12.130.111:8082/offer",
        {
          "Content-Type": "application/json",
        },
        JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        })
      );

      const answer = await response.json();
      if (pc.current) {
        await pc.current.setRemoteDescription(
          new RTCSessionDescription(answer)
        );
      }
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  const start = async (): Promise<void> => {
    const configuration = {
      sdpSemantics: "unified-plan",
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    };
    pc.current = new RTCPeerConnection(configuration);

    pc.current.ondatachannel = (event) => {  // 이 부분을 추가
        const channel = event.channel;
        channel.onmessage = (event) => {
          const receivedData = JSON.parse(event.data);
          const label = `Id: ${receivedData.id}, X: ${receivedData.label_x}, Y: ${receivedData.label_y}`;
          setDataLabel(label);
        };
      };

    try {
      const stream = await mediaDevices.getUserMedia({
        audio: true,
        video: {
          width: 3840,
          height: 2160,
          frameRate: 20,
          facingMode: "environment",
        },
      });
      stream.getTracks().forEach((track) => {
        pc.current?.addTrack(track, stream);
      });

      await negotiate();
      setIsConnected(true);
      setLocalStream(stream);
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  const stop = (): void => {
    setIsConnected(false);
    if (localStream) {
      // localStream의 모든 트랙을 종료
      localStream.getTracks().forEach((track) => {
        track.stop();
      });
    }
    setLocalStream(null);
    pc.current?.close();
  };

  useEffect(() => {
    start();
    // 컴포넌트가 언마운트될 때 스트림을 정지
    return () => {
      stop();
    };
  }, []);

  return (
    <View style={styles.container}>
      {localStream && (
        <RTCView
          style={styles.video}
          streamURL={localStream.toURL()}
          objectFit="cover"
        />
      )}
      {dataLabel && (
        <View style={styles.labelContainer}>
          <Text style={styles.labelText}>{dataLabel}</Text>
        </View>
      )}
      <View style={styles.buttonContainer}>
        <Button title="Start" onPress={start} disabled={isConnected} />
        <Button title="Stop" onPress={stop} disabled={!isConnected} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  video: {
    flex: 1, // 전체 화면 공간을 사용
    width: "100%",
    height: "100%",
  },
  buttonContainer: {
    position: "absolute",
    bottom: 10,
    left: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    width: "90%",
  },
  labelContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [{ translateX: -50% }, { translateY: -50% }],
    padding: 10,
    backgroundColor: "rgba(0,0,0,0.6)",
    borderRadius: 5,
  },
  labelText: {
    color: "white",
    fontSize: 14,
  },
});

export default App;