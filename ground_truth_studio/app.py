"""Run: python -m streamlit run ground_truth_studio/app.py (loopback only)."""
from __future__ import annotations

import atexit
import copy
import json
import logging
import sys
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from ground_truth_studio.adjudication_service import AdjudicationService
from ground_truth_studio.annotation_service import AnnotationService
from ground_truth_studio.config import Settings
from ground_truth_studio.export_service import ExportService, summary
from ground_truth_studio.models import Annotation
from ground_truth_studio.review_repository import ReviewRepository
from ground_truth_studio.single_reviewer import export_single_reviewer
from ground_truth_studio.video_service import MediaServer, VideoService
from ground_truth_studio.background_jobs import BackgroundJobs

LOGGER = logging.getLogger(__name__)
player = components.declare_component("gt_source_player", path=str(Path(__file__).parent / "player"))


@st.cache_resource
def preparation_jobs() -> BackgroundJobs:
    return BackgroundJobs()


def prepared(key: tuple[Any, ...], operation: Callable[[], Any]) -> Any:
    """Render preparation progress while bounded workers inspect/transcode sources."""
    future = preparation_jobs().request(key, operation)
    try:
        return future.result(timeout=.1)
    except FutureTimeout:
        @st.fragment(run_every=1)
        def progress():
            if future.done():
                st.rerun()
            st.info("PREPARING_SOURCE · Đang chuẩn bị video ở background; ngài có thể chuyển trang.")
        progress()
        return None


@st.cache_resource
def resources(repo: str, private: str):
    repository = ReviewRepository(Settings(Path(repo), Path(private)))
    videos = VideoService(repository.settings)
    atexit.register(videos.close)
    return repository, videos


@st.cache_resource
def media_server(repo: str, private: str):
    repository, _ = resources(repo, private)
    server = MediaServer(repository)
    atexit.register(server.close)
    return server


def overview(repository: ReviewRepository) -> None:
    metrics = summary(repository)
    st.subheader("Hồ sơ độc lập, bằng chứng có nguồn")
    st.caption("Nguồn gốc → Human review → Candidate PROVISIONAL; single-reviewer không phải kiểm định chéo")
    columns = st.columns(4)
    for column, (label, value) in zip(columns, [("Positive historical", metrics["positive_videos"]), ("VN_POS candidates", metrics["positive_candidate_videos"]),
                                               ("Negative candidates có nguồn", metrics["negative_candidates_available"]),
                                               ("Missing historical references", metrics["missing_historical_references"])]):
        column.metric(label, value)
    st.write({"Technical verified (audit hiện tại)": metrics["technical_verified_available"],
              "Human reviewed / submitted": metrics["submitted_reviews"],
              "Scientific eligible negative hours": metrics["eligible_negative_exposure_hours"]})
    left, right = st.columns([1.5, 1])
    with left, st.container(border=True):
        st.markdown("**Tiến độ thực tế trong Studio**")
        st.write({"Nguồn có trên đĩa": metrics["available_sources"], "Video chưa review trong Studio": metrics["videos_without_studio_review"],
                  "Review đã submit": metrics["submitted_reviews"], "Video có bất đồng": metrics["disputed_videos"],
                  "Video đã phân xử": metrics["adjudicated_videos"], "Candidate event được human adjudication": metrics["human_adjudicated_candidate_events"]})
    with right, st.container(border=True):
        st.markdown("**Trạng thái khoa học**")
        st.write(metrics["scientific_status"])
        st.caption(f"{metrics['historical_cross_verified_claims']} claim cross_verified lịch sử chưa được Studio xác thực lại. Candidate không thay thế Ground Truth frozen.")
        st.caption("Tài khoản được cấp theo người thật và vai trò. Người quản lý chịu trách nhiệm xác nhận danh tính độc lập.")


def review_page(repository: ReviewRepository, videos: VideoService, token: str, kind: str) -> None:
    own = {row["video_id"]: row for row in repository.own_reviews(token)}
    with st.sidebar:
        status = st.selectbox("Lọc trạng thái của tôi", ["ALL", "PENDING", "REVIEWED", "DISPUTED", "VERIFIED"])
        available = repository.working_videos(kind)
        split_hints = {}
        if kind == "positive_candidate":
            from ground_truth_studio.preannotation import split_metadata
            try:
                split_hints = split_metadata(repository.settings)
                available.sort(key=lambda row: (split_hints.get(row['video_id'], {}).get('annotation_priority', 4), row['video_id']))
            except (ValueError, OSError, KeyError) as error:
                st.warning(f"Manifest chia tập cần kiểm tra: {error}")
            st.caption("Chia tập trước gán nhãn là provisional. Ứng viên test chưa được xác minh độc lập; không chạy detector để chọn nhãn/ngưỡng.")
        st.caption(f"{len(available)} nguồn hiện có; references thiếu được giữ trong catalog, không đưa vào dropdown.")
        def label(row):
            review = own.get(row["video_id"])
            if review and review.get("adjudication_status") in {"DISPUTED", "ACCEPT"}:
                return "DISPUTED" if review["adjudication_status"] == "DISPUTED" else "VERIFIED"
            return "REVIEWED" if review and review["status"] == "SUBMITTED" else "PENDING"
        filtered = [row for row in available if status == "ALL" or label(row) == status]
        if not filtered:
            st.info("Chưa có bản ghi thuộc trạng thái này. VERIFIED trong Studio chỉ là human adjudicated candidate, chưa phải GT OFFICIAL.")
            return
        selected = st.selectbox("Video nguồn", [row["video_id"] for row in filtered], format_func=lambda vid: f"{vid} · {label(repository.video(vid))}" + (f" · {split_hints[vid]['proposed_split']}" if vid in split_hints else ""))
        if selected in split_hints:
            hint = split_hints[selected]
            if hint['source_sha256'] != repository.video(selected).get('expected_sha256'):
                st.error("Source pin khác manifest chia tập; cần đối soát trước review.")
                return
            st.caption(f"Nhóm cảnh: {hint['scene_group_id']} · {hint['candidate_readiness']} · {hint['independence_status']}")
            if hint.get('decode_status') != 'NORMAL_EOF':
                st.warning("Nguồn chưa đạt audit giải mã đầy đủ; không dùng để nghiệm thu test độc lập. Giữ nguyên video gốc.")
        if st.button("Mở / tiếp tục phiên review", type="primary"):
            review = repository.start_review(token, selected)
            st.session_state["active_review"] = review["id"]
            st.session_state.pop("editing_payload", None)
    if "active_review" not in st.session_state:
        st.info("Chọn video và mở phiên review. Nhãn, mốc thời gian của người khác và kết quả AI được ẩn.")
        return
    review = repository.review(token, st.session_state["active_review"])
    if review["video_id"] != selected:
        st.info("Nhấn Mở / tiếp tục để chuyển sang video đã chọn.")
        return
    video = repository.video(selected)
    from ground_truth_studio.independent_review import IndependentReview
    with st.expander("Đánh giá độc lập A/B — giữ nguyên nhãn chung cũ"):
        prior_joint = st.checkbox("Bản trước là JOINT_CONSENSUS_PRIOR", key="joint_" + review["id"])
        if st.button("Mở lượt độc lập trắng A/B"):
            fresh = IndependentReview(repository).start(token, selected, joint_prior=prior_joint)
            st.session_state["active_review"] = fresh["id"]
            st.session_state.pop("editing_review", None)
            st.rerun()
        st.caption("Chỉ thực hiện sau khi hai người đã có tài khoản/person_id riêng; không xem nhãn cũ, nhãn người kia hoặc detector trong lượt mới.")
        if st.button("Xuất hai lượt độc lập đã Submit"):
            try:
                st.success(str(IndependentReview(repository).export(token, selected)))
            except (PermissionError, ValueError) as error:
                st.error(str(error))
    st.subheader(f"{selected} · Review độc lập")
    st.caption(f"Nguồn catalog: {video['relative_path']} · Trạng thái: {review['status']} · Revision {review['revision']}")
    if review["status"] == "SUBMITTED":
        if st.button("Bắt đầu blind re-review của chính tôi"):
            fresh = repository.start_blind_rereview(token, selected)
            st.session_state["active_review"] = fresh["id"]
            st.session_state.pop("editing_review", None)
            st.rerun()
        st.success("Review đã khóa. Chỉ bản ghi của ngài được hiển thị ở đây; chờ phân xử hoặc yêu cầu review lại.")
        st.json(review["payload"])
        return
    try:
        from ground_truth_studio.config import confined
        source_path = confined(repository.settings.repo, video["relative_path"])
        stat = source_path.stat()
        # Stat is a cache invalidation hint only; SHA verification remains in VideoService/submission.
        metadata = prepared(("inspect", str(repository.settings.private), selected, video.get("expected_sha256"), stat.st_size, stat.st_mtime_ns), lambda: videos.inspect(video))
        if metadata is None:
            return
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    key = "frame_" + review["id"]
    st.session_state.setdefault(key, 0)
    if st.session_state.get("editing_review") != review["id"]:
        st.session_state["editing_review"] = review["id"]
        st.session_state["editing_payload"] = copy.deepcopy(review["payload"])
    payload = st.session_state["editing_payload"]
    left, right = st.columns([1.8, 1])
    with left:
        clock_key = "media_" + review["id"]
        if st.button("Bật trình phát video", help="AVI được tạo proxy MP4 riêng; video gốc không đổi. Không nạp toàn bộ video vào RAM."):
            st.session_state["play_requested_" + review["id"]] = True
        if st.session_state.get("play_requested_" + review["id"]):
            path = prepared(("playback", str(repository.settings.private), metadata["source_sha256"]), lambda: videos.playback_path(video, metadata))
            if path is None:
                return
            server = media_server(str(repository.settings.repo), str(repository.settings.private))
            st.session_state[clock_key] = server.grant(token, path)
            st.session_state.pop("play_requested_" + review["id"], None)
        if clock_key in st.session_state:
            pts = metadata["source_pts"]
            seek_time = pts[st.session_state[key]] - pts[0] if pts else st.session_state[key] / metadata["fps"]
            seek_time = videos.playback_time_from_source(metadata, seek_time)
            action = player(url=st.session_state[clock_key], seek_nonce=st.session_state[key], seek_time=seek_time, key="player_" + review["id"], default=None)
            if action and st.session_state.get("last_player_action") != action["nonce"]:
                st.session_state["last_player_action"] = action["nonce"]
                source_time = videos.source_time_from_playback(metadata, float(action["timestamp"]))
                st.session_state[key] = videos.index_at_time(metadata, source_time)
                if action["action"] in {"mark_start", "mark_end"}:
                    _, point = videos.frame(video, st.session_state[key])
                    st.session_state[review["id"] + action["action"]] = point
        st.number_input("Frame index thực tế (từ 0)", min_value=0, max_value=metadata["frame_count"] - 1, step=1, key=key)
        navigation = st.columns(3)
        def move(delta):
            st.session_state[key] = min(max(st.session_state[key] + delta, 0), metadata["frame_count"] - 1)
        navigation[0].button("← Previous Frame", on_click=move, args=(-1,))
        navigation[1].button("Next Frame →", on_click=move, args=(1,))
        seek = navigation[2].number_input("Tua đến giây", min_value=0.0, max_value=float(metadata["duration"]), value=0.0, step=.1)
        if st.button("Định vị timestamp"):
            st.session_state["seek_pending"] = (key, videos.index_at_time(metadata, seek))
            st.rerun()
        image, point = videos.frame(video, int(st.session_state[key]))
        st.image(image, channels="BGR", width="stretch")
        st.caption(f"Frame {point['frame_index']} / {metadata['frame_count']-1} · {point['timestamp_sec']:.6f}s · FPS metadata {metadata['fps']:.4f} · Duration {metadata['duration']:.4f}s")
        if metadata["timestamp_provenance"] != "SOURCE_PTS":
            st.warning("Timestamp chỉ được ước lượng từ frame index/FPS; chưa có source PTS. Không dùng làm chứng minh VFR timing hay FAR eligibility.")
        else:
            st.caption("Timestamp: SOURCE_PTS · Frame được decode theo ordinal nguồn, không seek bằng FPS giả định.")
    with right, st.container(border=True):
        st.markdown("**Gán interval của ngài**")
        choice = st.selectbox("Nhãn", ["FIRE", "SMOKE", "UNCERTAIN"])
        marks = st.columns(2)
        if marks[0].button("Mark Start"):
            st.session_state[review["id"] + "mark_start"] = point
        if marks[1].button("Mark End"):
            st.session_state[review["id"] + "mark_end"] = point
        start = st.session_state.get(review["id"] + "mark_start")
        end = st.session_state.get(review["id"] + "mark_end")
        st.caption(f"Onset: {start['timestamp_sec'] if start else '—'} · Offset: {end['timestamp_sec'] if end else '—'}")
        if st.button("Add Event"):
            if not start or not end:
                raise ValueError("Đánh dấu đủ hai frame onset/offset trước")
            event = Annotation.create(choice, start, end).to_dict()
            from ground_truth_studio.models import validate_events
            validate_events([event], metadata["duration"], uncertain=choice == "UNCERTAIN")
            payload["uncertain_intervals" if choice == "UNCERTAIN" else "events"].append(event)
        all_events = payload["events"] + payload["uncertain_intervals"]
        for event in all_events:
            st.write(f"{event['label']} · {event['start_sec']:.4f} → {event['end_sec']:.4f}s")
            edit, delete = st.columns(2)
            if edit.button("Edit", key="edit_" + event["event_id"]):
                st.session_state["edit_event"] = event["event_id"]
                st.info("Đánh dấu lại Start/End rồi nhấn Cập nhật interval.")
            if delete.button("Delete", key="delete_" + event["event_id"]):
                payload["events"] = [item for item in payload["events"] if item["event_id"] != event["event_id"]]
                payload["uncertain_intervals"] = [item for item in payload["uncertain_intervals"] if item["event_id"] != event["event_id"]]
                if st.session_state.get("edit_event") == event["event_id"]:
                    st.session_state.pop("edit_event", None)
        if st.session_state.get("edit_event") and st.button("Cập nhật interval"):
            if not start or not end:
                raise ValueError("Đánh dấu đủ hai frame trước")
            changed = Annotation.create(choice, start, end).to_dict()
            target = st.session_state.pop("edit_event")
            changed["event_id"] = target
            payload["events"] = [item for item in payload["events"] if item["event_id"] != target]
            payload["uncertain_intervals"] = [item for item in payload["uncertain_intervals"] if item["event_id"] != target]
            payload["uncertain_intervals" if choice == "UNCERTAIN" else "events"].append(changed)
        decisions = ["", "VERIFIED_NEGATIVE", "CONTAINS_FIRE", "CONTAINS_SMOKE", "UNCERTAIN"]
        if kind == "negative":
            payload["decision"] = st.selectbox("Quyết định semantic", decisions, index=decisions.index(payload.get("decision", "")))
        payload["evidence_ref"] = st.text_input("Evidence reference", value=payload.get("evidence_ref", ""), key="evidence_" + review["id"])
        payload["notes"] = st.text_area("Ghi chú của ngài", value=payload.get("notes", ""), key="notes_" + review["id"])
        payload["attest_full_view"] = st.checkbox("Tôi đã xem toàn bộ nguồn gốc và chịu trách nhiệm cho review này", value=payload.get("attest_full_view", False), key="attest_" + review["id"])
        payload["attest_independent"] = st.checkbox("Tôi đánh giá lượt này độc lập, không xem nhãn chung/nhãn người kia/detector", value=payload.get("attest_independent", False), key="independent_" + review["id"])
        service = AnnotationService(repository, videos)
        save, submit = st.columns(2)
        if save.button("Save Draft", type="primary"):
            service.save(token, review["id"], review["revision"], payload)
            st.success("Đã lưu bền vững bản nháp")
            st.rerun()
        if submit.button("Submit Review"):
            service.save(token, review["id"], review["revision"], payload, submit=True)
            st.rerun()
        if st.button("Nạp bản CSV của chính tài khoản này thành draft"):
            service.import_own_csv_draft(token, review["id"], review["revision"])
            st.session_state.pop("editing_review", None)
            st.rerun()


def adjudication_page(repository: ReviewRepository, videos: VideoService, token: str) -> None:
    st.subheader("Phân xử có bằng chứng")
    video_id = st.selectbox("Video cần phân xử", [row["video_id"] for row in repository.videos()])
    service = AdjudicationService(repository)
    context = service.comparison(token, video_id)
    first, second = st.columns(2)
    first.json(context["reviews"][0])
    second.json(context["reviews"][1])
    st.json(context["comparison"])
    with st.expander("Ground Truth lịch sử — claim chưa được tự xác thực", expanded=False):
        st.json(context["historical"])
    video = repository.video(video_id)
    metadata = videos.inspect(video)
    frame_index = st.number_input("Frame nguồn để đối soát", min_value=0, max_value=metadata["frame_count"] - 1, value=0)
    image, point = videos.frame(video, frame_index)
    st.image(image, channels="BGR", width="stretch")
    st.caption(f"{point['timestamp_sec']:.6f}s · {point['timestamp_provenance']}")
    decision = st.selectbox("Quyết định cuối", ["DISPUTED", "REQUIRE_REVIEW", "REJECT", "ACCEPT"])
    selected = st.radio("Nguồn interval cuối (không lấy trung bình)", ["Annotator 01", "Annotator 02", "Danh sách tự phân xử"])
    final = context["reviews"][0 if selected == "Annotator 01" else 1]["payload"]["events"]
    if selected == "Danh sách tự phân xử":
        final = json.loads(st.text_area("Final event JSON (giữ frame/source provenance)", value="[]"))
    reason = st.text_area("Lý do phân xử")
    evidence = st.text_input("Bằng chứng phân xử")
    omissions = json.loads(st.text_area("Explicit disagreement cho event lịch sử bị thiếu: JSON {GT11-id: lý do}", value="{}"))
    if st.button("Ghi quyết định phân xử", type="primary"):
        service.decide(token, video_id, decision, reason, evidence, final, omissions)
        st.success("Đã lưu quyết định và provenance; không thay snapshot frozen")


def main() -> None:
    st.set_page_config(page_title="Ground Truth Review Studio", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>.stApp{background:#0b121a;color:#dbe7ef}section[data-testid=stSidebar]{background:#111f2a}h1,h2,h3,p,label{color:inherit!important}input,textarea,[data-baseweb=select]>div{background:#182a35!important;color:#e5f1ee!important}div[data-testid=stMetric]{border:1px solid #26423d;border-radius:12px;padding:16px;background:#10231f}button[kind=primary]{background:#207a5b}header[data-testid=stHeader]{background:#0b121a}div[data-testid=stForm]{border-color:#284039}</style>""", unsafe_allow_html=True)
    st.caption("DACN / WEEK 4 / HUMAN ANNOTATION")
    st.title("Ground Truth Review Studio")
    st.caption("Nguồn gốc rõ ràng. Phiên review độc lập. Quyết định thuộc về con người.")
    settings = Settings.load()
    repository, videos = resources(str(settings.repo), str(settings.private))
    if "seek_pending" in st.session_state:
        key, index = st.session_state.pop("seek_pending")
        st.session_state[key] = index
    token = st.session_state.get("auth_token")
    if not token:
        left, right = st.columns([1.5, 1])
        with left:
            overview(repository)
        with right, st.form("login"):
            st.subheader("Đăng nhập reviewer")
            reviewer = st.text_input("Reviewer ID")
            password = st.text_input("Mật khẩu", type="password")
            if st.form_submit_button("Đăng nhập", type="primary"):
                st.session_state["auth_token"] = repository.login(reviewer, password)
                st.rerun()
            st.caption("Tài khoản do người quản lý cấp theo danh tính thực tế. Không có tài khoản mặc định hoặc tự đổi vai trò.")
        return
    actor = repository.actor(token)
    with st.sidebar:
        st.markdown("### REVIEW WORKSPACE")
        st.write(actor["id"])
        st.caption(actor["role"])
        pages = ["Overview"]
        if actor["role"].startswith("annotator_"):
            pages += ["Positive Review", "VN_POS Candidate Review", "Negative Review", "Single Reviewer Export", "Review Progress"]
        if actor["role"] == "adjudicator":
            pages += ["Adjudication"]
        if actor["role"] in {"adjudicator", "admin"}:
            pages += ["Export Report"]
        page = st.radio("Điều hướng", pages)
        if st.button("Đăng xuất"):
            repository.logout(token)
            videos.close()
            st.session_state.clear()
            st.rerun()
    if page == "Overview":
        overview(repository)
    elif page.endswith("Review"):
        review_page(repository, videos, token, "positive_candidate" if page.startswith("VN_POS") else "positive" if page.startswith("Positive") else "negative")
    elif page == "Review Progress":
        st.subheader("Tiến độ của ngài — không phải independent cross-verification")
        st.dataframe([{"video_id": row["video_id"], "status": row["status"], "revision": row["revision"], "submitted_at_utc": row["submitted_at"]}
                      for row in repository.own_reviews(token)])
    elif page == "Single Reviewer Export":
        st.subheader("Single-Reviewer Research Mode")
        st.warning("Dữ liệu một người đánh giá: PROVISIONAL. Không phải CROSS_VERIFIED, không phải FAR/Recall/Delay chính thức.")
        own = repository.own_review_history(token)
        submitted = [r for r in own if r["status"] == "SUBMITTED"]
        st.metric("Video đã Submit", len(submitted))
        st.metric("Video còn Draft", len([r for r in own if r["status"] == "DRAFT"]))
        st.caption("Dùng Positive/Negative Review như hiện tại. Save Draft để lưu; Submit để khóa review. Không cần tạo tài khoản giả thứ hai.")
        if st.button("Xuất kết quả một người (PROVISIONAL)", disabled=not submitted, type="primary"):
            destination = export_single_reviewer(repository, token)
            st.success(f"Đã tạo bundle riêng, không sửa CSV hay GT frozen: {destination}")
    elif page == "Adjudication":
        adjudication_page(repository, videos, token)
    else:
        st.subheader("Candidate export & provenance")
        st.write(summary(repository))
        conflicts = repository.snapshot_csv()
        if conflicts:
            st.error("CSV thay đổi bên ngoài, giữ nguyên SQLite và CSV: " + ", ".join(conflicts))
        if st.button("Xuất bundle mới trong private directory", disabled=bool(conflicts), type="primary"):
            destination = ExportService(repository).export(token)
            st.success(str(destination))
            st.caption("Candidate chưa phải Ground Truth OFFICIAL. CSV gốc và snapshot không đổi.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError, OSError) as error:
        LOGGER.info("Studio operation blocked: %s", type(error).__name__)
        st.error(str(error))
