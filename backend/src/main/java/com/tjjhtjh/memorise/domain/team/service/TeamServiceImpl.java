package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.tag.repository.TaggedTeamRepository;
import com.tjjhtjh.memorise.domain.team.exception.*;
import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.repository.TeamUserRepository;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;
import com.tjjhtjh.memorise.domain.team.service.dto.request.*;
import com.tjjhtjh.memorise.domain.team.service.dto.response.*;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TeamServiceImpl implements TeamService {

    private final TeamRepository teamRepository;
    private final UserRepository userRepository;
    private final TeamUserRepository teamUserRepository;
    private final TaggedTeamRepository taggedTeamRepository;

    private static final String NO_USER = "회원 정보가 존재하지 않습니다";
    private static final String NO_TEAM = "팀 정보가 존재하지 않습니다";
    private static final String NOT_MEMBER = "그룹의 멤버가 아닙니다";
    private static final String NO_AUTHORITY = "권한이 없습니다";
    private static final String EXISTED_MEMBER = "이미 그룹에 속해있는 멤버입니다";
    private static final String WRONG_REQUEST = "잘못된 요청입니다";

    @Transactional
    public CreateTeamResponse createTeam(CreateTeamRequest createTeamRequest) {
        User owner = userRepository.findByUserSeqAndIsDeletedFalse(createTeamRequest.getOwner()).orElseThrow(() -> new NoUserException(NO_USER));
        Team team = (createTeamRequest.getPassword().equals("")) ? new Team(createTeamRequest.getName(), owner.getUserSeq()) : new Team(createTeamRequest.getName(), owner.getUserSeq(), createTeamRequest.getPassword());
        teamRepository.save(team);
        teamUserRepository.save(new TeamUser(team, owner));
        return new CreateTeamResponse(true, team.getTeamSeq());
    }

    @Override
    public TeamDetailResponse getTeamDetailInfo(Long teamSeq, Long userSeq) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));

        UserInfoResponse me = new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER)));
        UserInfoResponse owner = team.getOwner().equals(userSeq) ? null : new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(team.getOwner()).orElseThrow(() -> new NoUserException(NO_USER)));

        List<Long> userSeqs = teamUserRepository.findAllUserByTeamSeq(teamSeq);
        if (!userSeqs.contains(userSeq)) {
            throw new NotMemberOfGroup(NOT_MEMBER);
        }
        List<UserInfoResponse> members = new ArrayList<>();
        for(Long user : userSeqs) {
            if (!user.equals(userSeq) && !user.equals(team.getOwner())) {
                members.add(new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(user).orElseThrow(() -> new NoUserException(NO_USER))));
            }
        }

        return new TeamDetailResponse(team, me, owner, members);
    }

    @Override
    public List<TeamListResponse> getTeamList(Long userSeq, String keyword) {
        User me = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        List<Team> teams = teamRepository.findAllByContainsKeyword(userSeq, keyword);

        List<TeamListResponse> teamListResponses = new ArrayList<>();
        for (Team team : teams) {
            User owner = userRepository.findByUserSeqAndIsDeletedFalse(team.getOwner()).orElseThrow(() -> new NoUserException(NO_USER));
            List<String> profiles = teamRepository.findUserProfiles(team.getTeamSeq(), userSeq);
            boolean isParticipated = teamUserRepository.findByTeamSeqAndUserSeq(team.getTeamSeq(), userSeq) != null;
            if (userSeq == team.getOwner()) {
                profiles.add(0, owner.getProfile());
            } else if (isParticipated) {
                profiles.add(0, me.getProfile());
                profiles.add(1, owner.getProfile());
            } else {
                profiles.add(0, owner.getProfile());
            }
            teamListResponses.add(new TeamListResponse(team, profiles.subList(0, Math.min(profiles.size(), 4)), isParticipated));
        }
        return teamListResponses;
    }

    @Override
    @Transactional
    public InviteMemberResponse inviteMember(Long teamSeq, InviteMemberRequest inviteMemberRequest) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
        if(!team.getOwner().equals(inviteMemberRequest.getUserSeq())) {
            throw new NoAuthorityException(NO_AUTHORITY);
        }
        if(teamUserRepository.findAllUserByTeamSeq(teamSeq).contains(inviteMemberRequest.getTargetSeq())) {
            throw new ExistedMemberException(EXISTED_MEMBER);
        }
        User user = userRepository.findByUserSeqAndIsDeletedFalse(inviteMemberRequest.getTargetSeq()).orElseThrow(() -> new NoUserException(NO_USER));
        teamUserRepository.save(new TeamUser(team, user));
        return new InviteMemberResponse(true);
    }

    @Override
    public List<InviteUserListResponse> getInviteUserList(Long teamSeq, Long userSeq, String keyword) {
        if(!teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM)).getOwner().equals(userSeq)) {
            throw new NoAuthorityException(NO_AUTHORITY);
        }
        return teamRepository.findInviteUserList(teamSeq, userSeq, keyword);
    }

    @Override
    @Transactional
    public void kickMember(Long teamSeq, KickMemberRequest kickMemberRequest) {
        if (kickMemberRequest.getUserSeq().equals(kickMemberRequest.getTargetSeq())) {
            throw new WrongRequestException(WRONG_REQUEST);
        }
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
        if (!team.getOwner().equals(kickMemberRequest.getUserSeq())) {
            throw new NoAuthorityException(NO_AUTHORITY);
        }
        if (!teamUserRepository.findAllUserByTeamSeq(teamSeq).contains(kickMemberRequest.getTargetSeq())) {
            throw new NotMemberOfGroup(NOT_MEMBER);
        }
        TeamUser teamUser = teamUserRepository.findByTeamSeqAndUserSeq(teamSeq, kickMemberRequest.getTargetSeq());
        teamUserRepository.delete(teamUser);
    }

    @Override
    public List<MyTeamListResponse> getMyTeamList(Long userSeq) {
        User me = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        List<Long> teamSeqs = teamUserRepository.findAllTeamByUserSeq(userSeq);

        List<MyTeamListResponse> myTeamListResponses = new ArrayList<>();
        for (Long teamSeq : teamSeqs) {
            Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
            List<Long> userSeqs = teamUserRepository.findAllUserByTeamSeq(teamSeq);
            List<String> memberProfiles = new ArrayList<>();
            memberProfiles.add(userRepository.findByUserSeqAndIsDeletedFalse(team.getOwner()).orElseThrow(() -> new NoUserException(NO_USER)).getProfile());
            for (Long user : userSeqs) {
                if (memberProfiles.size() == 3) break;
                if (!user.equals(userSeq) && !user.equals(team.getOwner())) {
                    memberProfiles.add(userRepository.findByUserSeqAndIsDeletedFalse(user).orElseThrow(() -> new NoUserException(NO_USER)).getProfile());
                }
            }
            myTeamListResponses.add(new MyTeamListResponse(team, me, memberProfiles));
        }
        return myTeamListResponses;
    }

    @Override
    @Transactional
    public EnterTeamResponse enterTeam(Long teamSeq, EnterTeamRequest enterTeamRequest) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
        if (team.getPassword() != null && !team.getPassword().equals(enterTeamRequest.getPassword())) {
            throw new WrongCodeException(WRONG_REQUEST);
        }
        teamUserRepository.save(new TeamUser(team, userRepository.findByUserSeqAndIsDeletedFalse(enterTeamRequest.getUserSeq()).orElseThrow(() -> new NoUserException(NO_USER))));
        return new EnterTeamResponse(true);
    }

    @Override
    @Transactional
    public ExitTeamResponse exitTeam(Long teamSeq, Long userSeq) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
        if (team.getOwner().equals(userSeq)) {
            for (Long memberSeq: teamUserRepository.findAllUserByTeamSeq(teamSeq)) {
                teamUserRepository.delete(teamUserRepository.findByTeamSeqAndUserSeq(teamSeq, memberSeq));
            }
            taggedTeamRepository.deleteAllByTeamSeq(teamSeq);
            teamRepository.delete(team);
        } else {
            teamUserRepository.delete(teamUserRepository.findByTeamSeqAndUserSeq(teamSeq, userSeq));
        }
        return new ExitTeamResponse(true);
    }

    @Override
    @Transactional
    public UpdateTeamResponse updateTeam(Long teamSeq, UpdateTeamRequest updateTeamRequest) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));
        if (!team.getOwner().equals(updateTeamRequest.getUserSeq())) {
            throw new NoAuthorityException(NO_AUTHORITY);
        }
        team.update(updateTeamRequest.getName(), updateTeamRequest.getPassword());
        return new UpdateTeamResponse(true);
    }
}
