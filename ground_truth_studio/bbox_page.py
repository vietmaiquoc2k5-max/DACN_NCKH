"""Human bbox workspace using the existing Studio session and private storage."""
import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import cv2

from ground_truth_studio.bbox_service import BBoxReviewService, validate_boxes
from ground_truth_studio.bbox_queue import describe_queue, filter_queue, paginate

canvas = components.declare_component('human_bbox_canvas',path=str(Path(__file__).parent/'bbox_canvas'))


def bbox_page(repository, token: str) -> None:
    """Display/edit real boxes, with explicit approval and separate lineage gate."""
    service = BBoxReviewService(repository)
    history = {r['image_sha256']:r for r in service.history(token)}
    issues=getattr(service,'integrity_issues',[])
    if issues:
        st.warning(f'{len(issues)} version đang INTEGRITY_QUARANTINED; chưa tính là approval và export bị chặn. Các review hợp lệ khác vẫn dùng được.')
        with st.expander('Chi tiết integrity quarantine'):
            st.dataframe([{'version_id':i['version_id'],'error':i['error'],'path':i['path']} for i in issues])
    st.subheader('Human bbox QA · 0 SMOKE / 1 FIRE')
    st.caption('Phê duyệt bbox riêng với event GT; chưa được phép training khi lineage chưa xác minh.')
    st.write({'Ảnh trong queue đã loại trùng':len(service.queue),'Đã xử lý bởi ngài':len(history),
              'Còn chờ':len(service.queue)-len(history),'Được phép training':0})
    rows = describe_queue(service.queue, history, service.repo)
    st.caption('Các trạng thái review dưới đây thuộc tài khoản đang đăng nhập; approval không tự xác minh lineage.')
    st.write({status:sum(r['review_status']==status for r in rows) for status in ('PENDING','APPROVED','REJECTED')})
    view = st.selectbox('Nhóm dữ liệu', ['Tất cả', 'Suspicious empty labels', 'Needs data investigation'], key='bbox_view')
    priority = st.checkbox('Priority', key='bbox_priority')
    empty = st.checkbox('Empty Label', key='bbox_empty')
    suspicious = st.checkbox('Suspicious Negative', key='bbox_suspicious')
    status = st.selectbox('Trạng thái review', ['ALL','PENDING','APPROVED','REJECTED'], key='bbox_status')
    query = st.text_input('Tìm Image ID / Video ID / SHA / path', key='bbox_search')
    filtered = filter_queue(rows, query=query, priority=priority, empty=empty,
        suspicious=suspicious or view=='Suspicious empty labels', status=status,
        investigation=view=='Needs data investigation')
    st.caption(f"Nhóm nghi vấn từ manifest: {sum(r['suspicious_negative'] for r in rows)} · Label hiện còn rỗng trong nhóm: {sum(r['suspicious_negative'] and r['empty_label'] for r in rows)} · Kết quả bộ lọc: {len(filtered)} / {len(rows)}")
    size = st.selectbox('Ảnh mỗi trang', [25,50,100], key='bbox_page_size')
    _, page, pages = paginate(filtered, int(st.session_state.get('bbox_page_number',1)), size)
    st.session_state['bbox_page_number'] = page
    page = int(st.number_input('Trang', min_value=1, max_value=pages, step=1, key='bbox_page_number'))
    subset, _, _ = paginate(filtered, page, size)
    identities = [r['image_sha256'] for r in subset]
    if identities:
        identity = st.selectbox('Ảnh cần review',identities,format_func=lambda k:service.queue[k].get('image_name',k), key='bbox_selected')
        metadata = next(r for r in subset if r['image_sha256']==identity)
        st.caption(f"ID: {metadata['image_id']} · Video: {metadata.get('video_id','UNKNOWN')} · {metadata['review_status']}")
        st.code(metadata['image_path'])
        st.caption('Lý do queue: '+metadata['queue_reason'])
        if metadata['investigation_reasons']:
            st.warning('Needs data investigation: '+', '.join(metadata['investigation_reasons'])+'. Có thể lưu quyết định bbox; chưa đủ điều kiện đưa ảnh vào training.')
        try:
            row,image,initial = service.inspect(token,identity)
        except (ValueError, OSError) as error:
            st.error(f'Không đọc được dữ liệu: {error}. Review đã lưu được giữ nguyên.')
            return
        if identity in history:
            initial = history[identity]['boxes']
        st.caption(f"{row['reserved_split']} · LINEAGE_NOT_VERIFIED · {row['image_name']}")
        if not initial:
            st.info('Label hiện có rỗng hoặc thiếu. Xem ảnh để xác nhận âm tính hoặc vẽ box mới.')
        if row.get('original_label_error'):
            st.warning('Nhãn gốc không hợp lệ; cần vẽ/chỉnh lại hoặc từ chối. Không tự coi ảnh là âm tính.')
        value = canvas(image='data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode(),
                       identity=identity,boxes=initial,key='bbox_'+identity,default=None)
        boxes = value['boxes'] if isinstance(value,dict) and value.get('identity')==identity else initial
        st.caption('Kéo chuột để vẽ. Chọn box để đổi class hoặc xóa; bảng bên dưới cho phép sửa tọa độ normalized.')
        revision=hashlib.sha256(json.dumps(boxes,sort_keys=True).encode()).hexdigest()[:12]
        edited = st.data_editor(pd.DataFrame(boxes,columns=['class_id','x','y','w','h']),num_rows='dynamic',key='table_'+identity+revision,
                    column_order=['class_id','x','y','w','h'])
        current=edited.to_dict('records')
        try:
            validate_boxes(current)
            preview=cv2.imread(str(image));height,width=preview.shape[:2]
            for box in current:
                x,y,w,h=(float(box[k]) for k in ['x','y','w','h'])
                left,top=int((x-w/2)*width),int((y-h/2)*height)
                right,bottom=int((x+w/2)*width),int((y+h/2)*height)
                color=(255,200,80) if box['class_id']==0 else (60,70,255)
                cv2.rectangle(preview,(left,top),(right,bottom),color,2)
                cv2.putText(preview,'SMOKE' if box['class_id']==0 else 'FIRE',(left,max(15,top)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
            st.image(preview,channels='BGR',caption='Xem trước chính xác nhãn sẽ lưu sau khi sửa bảng',width='stretch')
        except ValueError:
            st.warning('Tọa độ/class chưa hợp lệ; sửa bảng trước khi phê duyệt.')
        evidence = st.text_input('Bằng chứng / ghi chú quyết định',key='evidence_'+identity)
        attest = st.checkbox('Tôi đã xem toàn bộ ảnh và kiểm từng bbox',key='attest_'+identity)
        negative = st.checkbox('Tôi xác nhận ảnh thực sự âm tính (không có box)',key='negative_'+identity)
        left,right = st.columns(2)
        decision = 'APPROVE' if left.button('Phê duyệt phiên bản bbox',type='primary') else 'REJECT' if right.button('Từ chối ảnh/nhãn') else None
        if decision:
            try:
                service.submit(token,identity,current,decision,evidence,attest,negative)
            except (ValueError, OSError) as error:
                st.error(f'Chưa lưu: {error}')
            else:
                st.success('Đã lưu annotation version mới trong private storage; nhãn gốc giữ nguyên.')
                st.rerun()
    if st.button('Xuất bbox đã phê duyệt của tôi'):
        try:st.success(str(service.export(token)))
        except (ValueError,OSError) as error:st.error(f'Export bị chặn: {error}')
