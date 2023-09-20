// 라이브러리
import axios from "axios";
import React, { useState } from "react";
import { View, Text, Image, TextInput } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";
import { useDispatch, useSelector } from "react-redux";

//컴포넌트
import ConfirmBtn from "../../components/Button/ConfirmBtn";
import HighlightHeader from "../../components/Header/HighlightHeader";
import ProfilePic from "../../components/ProfilePic";
import { RootStackParamList } from "../../App";
import { styles } from "./UserInputStyle";

//리듀서
import { setNickname, setProfileImg } from "../../store/user";

type SignUpScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  "SignUp"
>;

type Props = {
  navigation: SignUpScreenNavigationProp;
};

const SignUpScreen: React.FC<Props> = ({ navigation }) => {
  const dispatch = useDispatch();

  const userEmail = useSelector((state: any) => state.userInfo.email);

  const [userNickname, setUserNickname] = useState("");
  const [userProfileImg, setUserProfileImg] = useState("");

  const SignUpHandler = () => {
    //리덕스 정보 저장
    dispatch(setNickname(userNickname));

    // 백엔드 연동

    //메인페이지 이동
    navigation.navigate("Main");
  };

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <HighlightHeader />
      <View style={styles.container}>
        <View>
          <View style={{ alignItems: "center" }}>
            <View style={styles.profilebox}>
              <ProfilePic />
              <Image
                source={require("../../assets/icons/album.png")}
                style={styles.album}
              />
            </View>
          </View>
          <View style={styles.inputBox}>
            <Text style={styles.text}>닉네임</Text>
            <TextInput
              style={styles.input}
              value={userNickname}
              onChangeText={setUserNickname}
            />
            <Text style={styles.infoText}>
              한글, 영어, 숫자만 사용할 수 있어요. (최대 10자)
            </Text>
          </View>
        </View>

        <View style={styles.buttonContainer}>
          <ConfirmBtn onPress={SignUpHandler}>회원가입</ConfirmBtn>
        </View>
      </View>
    </LinearGradient>
  );
};

export default SignUpScreen;
