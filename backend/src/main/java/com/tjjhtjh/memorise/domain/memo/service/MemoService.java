package com.tjjhtjh.memorise.domain.memo.service;

import com.tjjhtjh.memorise.domain.memo.exception.BookMarkException;
import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.repository.BookmarkRepository;
import com.tjjhtjh.memorise.domain.memo.repository.MemoRepository;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.BookmarkRequest;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.MemoRequest;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class MemoService {

    private final MemoRepository memoRepository;
    private final UserRepository userRepository;
    private final BookmarkRepository bookMarkRepository;
    private static final String NO_USER_EMAIL = "이메일에 해당하는 유저가 없습니다";
    private static final String NO_MEMO = "해당하는 메모가 없습니다";
    private static final String NO_FIND_BOOKMARK = "해당하는 북마크를 찾을 수 없습니다";

    @Transactional
    public void createMemo(MemoRequest memoRequest){
       User user = userRepository.findByEmail(memoRequest.getUserId())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));

       memoRepository.save(memoRequest.registToEntity(user));
    }

    @Transactional
    public void updateMemo(MemoRequest memoRequest, Long memoId) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByEmail(memo.getUser().getEmail())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));

        memoRepository.save(memoRequest.updateToEntity(memoId, memoRequest,user));
    }

    @Transactional
    public void fakeDeleteMemo(Long memoId, MemoRequest memoRequest) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByEmail(memo.getUser().getEmail())
                        .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));

        String email = user.getEmail();

        memoRepository.save(memoRequest.deleteToEntity(memo,user));
        deleteBookmark(memoId,email);
    }

    @Transactional
    public void addBookmark(Long memoId, String email) throws MemoException {
            Memo memo = memoRepository.findById(memoId).orElseThrow(() -> new MemoException(NO_MEMO));
            User user = userRepository.findByEmail(email).orElseThrow(() -> new UsernameNotFoundException(NO_USER_EMAIL));

            bookMarkRepository.save(BookmarkRequest.saveToEntity(memo,user));
    }

    @Transactional
    public void deleteBookmark(Long memoId, String email){
            Bookmark bookmark = bookMarkRepository.findByMemoAndUser(memoId,email)
                    .orElseThrow(() -> new BookMarkException(NO_FIND_BOOKMARK));

            bookMarkRepository.delete(bookmark);
    }
}