package com.tjjhtjh.memorise.domain.team.repository.entity;

import static com.querydsl.core.types.PathMetadataFactory.*;

import com.querydsl.core.types.dsl.*;

import com.querydsl.core.types.PathMetadata;
import javax.annotation.processing.Generated;
import com.querydsl.core.types.Path;
import com.querydsl.core.types.dsl.PathInits;


/**
 * QTeamUser is a Querydsl query type for TeamUser
 */
@Generated("com.querydsl.codegen.DefaultEntitySerializer")
public class QTeamUser extends EntityPathBase<TeamUser> {

    private static final long serialVersionUID = -1179956252L;

    private static final PathInits INITS = PathInits.DIRECT2;

    public static final QTeamUser teamUser = new QTeamUser("teamUser");

    public final NumberPath<Long> GroupUserSeq = createNumber("GroupUserSeq", Long.class);

    public final QTeam team;

    public final com.tjjhtjh.memorise.domain.user.repository.entity.QUser user;

    public QTeamUser(String variable) {
        this(TeamUser.class, forVariable(variable), INITS);
    }

    public QTeamUser(Path<? extends TeamUser> path) {
        this(path.getType(), path.getMetadata(), PathInits.getFor(path.getMetadata(), INITS));
    }

    public QTeamUser(PathMetadata metadata) {
        this(metadata, PathInits.getFor(metadata, INITS));
    }

    public QTeamUser(PathMetadata metadata, PathInits inits) {
        this(TeamUser.class, metadata, inits);
    }

    public QTeamUser(Class<? extends TeamUser> type, PathMetadata metadata, PathInits inits) {
        super(type, metadata, inits);
        this.team = inits.isInitialized("team") ? new QTeam(forProperty("team")) : null;
        this.user = inits.isInitialized("user") ? new com.tjjhtjh.memorise.domain.user.repository.entity.QUser(forProperty("user")) : null;
    }

}

