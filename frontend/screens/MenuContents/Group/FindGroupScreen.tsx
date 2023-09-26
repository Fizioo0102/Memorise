import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import axios from "axios";
import { useEffect, useState } from "react";
import { View, Text, FlatList, Pressable, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";

import SmallBtn from "../../../components/Button/SmallBtn";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import SeachInput from "../../../components/SearchInput";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import { styles } from "./GroupStyle";
import GroupBox from "../../../components/GroupBox";
import AlertModal from "../../../components/Modal/AlertModal";

type RootStackParamList = {
  MakeGroup: undefined;
  GroupDetail: GroupDetailParams;
};

type GroupDetailParams = {
  teamSeq: number;
  userSeq: number;
};

type GroupData = [
  {
    teamSeq: number;
    teamName: string;
    myProfile: string;
    ownerProfile: string | null;
    memberProfiles: [string];
    password: boolean;
    owner: boolean;
    participated: boolean;
  }
];

const FindGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const MakeGroupHandler = () => navigation.navigate("MakeGroup");
  const userId = useSelector((state: RootState) => state.userInfo.id);
  const [groupData, setGroupData] = useState<GroupData | null>(null);

  // 그룹 참가 확인 모달
  const [isAlertModalVisible, setAlertModalVisible] = useState(false);
  const [targetTeamSeq, setTargetTeamSeq] = useState(0);
  const [targetTeamPassword, setTargetTeamPassword] = useState(false);
  const [targetTeamName, setTargetTeamName] = useState("");
  const [groupPassword, setGroupPassword] = useState("");

  const openAlertModal = (teamSeq: number, password: boolean) => {
    setAlertModalVisible(true);
    setTargetTeamSeq(teamSeq);
    setTargetTeamPassword(password);
  };

  // 취소 버튼 눌렀을 때
  const closeAlertModal = () => {
    setAlertModalVisible(false);
  };

  // 확인 버튼 눌렀을 때
  const joinConfirmHandler = () => {
    if (targetTeamPassword) {
      // 비공개 그룹
      // 비밀번호 입력 모달
    } else {
      // 공개 그룹
      // 그룹 참가
      joinGroupHandler(targetTeamSeq, "");
    }
  };

  // 그룹 참여 axios
  const joinGroupHandler = (teamSeq: number, password: string) => {
    axios({
      method: "POST",
      url: BACKEND_URL + `/teams/${teamSeq}`,
      data: {
        userSeq: userId,
        password: password,
      },
    })
      .then((res) => {
        console.log(res);
        // 디테일로
        navigation.navigate("GroupDetail", {
          teamSeq: teamSeq,
          // userSeq: userId,
          userSeq: 30, // 더미 데이터
        });
      })
      .catch((err) => {
        console.log(err);
      });
  };

  // 그룹 터치 시
  const onPressGroupHandler = (
    teamSeq: number,
    participated: boolean,
    password: boolean,
    teamName: string
  ) => {
    // 이미 참여 중인 그룹일 때
    if (participated) {
      navigation.navigate("GroupDetail", {
        teamSeq: teamSeq,
        // userSeq: userId,
        userSeq: 30, // 더미 데이터
      });
    } else {
      setTargetTeamName(teamName);
      openAlertModal(teamSeq, password);
    }
  };

  // 전체 그룹 목록 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          // url: BACKEND_URL + `/teams/${userId}`,
          url: BACKEND_URL + `/teams/30`, // 더미 데이터
        });
        setGroupData(res.data);
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

  console.log(groupData);

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>그룹 찾기</Text>
          <SmallBtn onPress={MakeGroupHandler}>만들기</SmallBtn>
        </View>
        <View style={{ marginTop: calculateDynamicWidth(25) }}>
          {/* <SeachInput /> */}
        </View>
        {groupData && (
          <FlatList
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
            data={groupData}
            keyExtractor={(item) => item.teamSeq.toString()}
            renderItem={({ item }) => (
              <View>
                <GroupBox
                  teamName={item.teamName}
                  myProfile={item.myProfile}
                  memberProfiles={item.memberProfiles}
                  ownerProfile={item.ownerProfile}
                  owner={item.owner}
                  goDetailHandler={() =>
                    onPressGroupHandler(
                      item.teamSeq,
                      item.participated,
                      item.password,
                      item.teamName
                    )
                  }
                  teamSeq={item.teamSeq}
                />
                {item.password && (
                  <View style={styles.lockIconContainer}>
                    <Image
                      style={styles.lockIcon}
                      source={require("../../../assets/image/lock.png")}
                    />
                  </View>
                )}
                {/* 그룹 참가 확인 모달 */}
                {isAlertModalVisible && (
                  <AlertModal
                    modalVisible={isAlertModalVisible}
                    closeModal={closeAlertModal}
                    onConfirm={joinConfirmHandler}
                    contentText={`${targetTeamName} 에\n참가하시겠습니까?`}
                    btnText="확인"
                  />
                )}
              </View>
            )}
          />
        )}
      </View>

      {/* 그룹 참가 확인 모달 */}
      {}
    </LinearGradient>
  );
};

export default FindGroupScreen;
