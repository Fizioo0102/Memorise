import axios from "axios";
import React, { useEffect, useState } from "react";
import { View, Text, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { RouteProp } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { BACKEND_URL } from "../../../util/http";
import { RootStackParamList } from "../../../App";
import { styles } from "./GroupStyle";

type GroupDetailScreenProps = {
  route: RouteProp<RootStackParamList, "GroupDetail">;
  navigation: StackNavigationProp<RootStackParamList, "GroupDetail">;
};

type MemoData = {
  name: string;
  me: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  owner: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  members: [
    {
      userSeq: number;
      email: string;
      nickname: string;
      profile: string;
    }
  ];
  password: string;
};

const GroupDetailScreen: React.FC<GroupDetailScreenProps> = ({
  route,
  navigation,
}) => {
  const { teamSeq, userSeq } = route.params;
  const [memoData, setMemoData] = useState<MemoData | null>(null);

  // 그룹 상세 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/teams/${teamSeq}/${userSeq}`,
        });
        setMemoData(res.data);
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, [memoData]);

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        {memoData && !memoData.owner ? (
          <>
            <View style={styles.titleContainer}>
              <Text style={styles.title}>{memoData.name}</Text>
              <Pressable style={styles.settingContainer}>
                <Image
                  style={styles.settingIcon}
                  source={require("../../../assets/icons/setting_sm.png")}
                />
                <Text style={styles.settingText}>그룹 설정</Text>
              </Pressable>
            </View>
            <View style={styles.memberContainer}>
              <Pressable style={styles.settingContainer}>
                <View style={styles.memberImageContainer}>
                  <Image
                    style={styles.memberImage}
                    source={require("../../../assets/image/addmember.png")}
                  />
                </View>
                <Text style={styles.memberText}>초대하기</Text>
              </Pressable>
              <Pressable style={styles.settingContainer}>
                <View>
                  <View style={styles.memberImageContainer}>
                    <Image
                      style={styles.memberImage}
                      source={require("../../../assets/image/profile_bg_square.png")}
                    />
                  </View>
                  <Text style={styles.memberText}>{memoData.me.nickname}</Text>
                </View>
              </Pressable>
            </View>
          </>
        ) : null}
      </View>
    </LinearGradient>
  );
};

export default GroupDetailScreen;
