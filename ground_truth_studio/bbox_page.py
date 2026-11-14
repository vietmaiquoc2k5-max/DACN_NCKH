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

canvas = components.declare_component('human_bbox_canvas',path=str(Path(__file__).parent/'bbox_canvas'))


def bbox_page(repository, token: str) -> None:
    """Display/edit real boxes, with explicit approval and separate lineage gate."""
    service = BBoxReviewService(repository)
    history = {r['image_sha256']:r for r in service.history(token)}
    st.subheader('Human bbox QA · 0 SMOKE / 1 FIRE')
    st.caption('Phê duyệt bbox riêng với event GT; chưa được phép training khi lineage chưa xác minh.')
    st.write({'Ảnh trong queue đã loại trùng':len(service.queue),'Đã xử lý bởi ngài':len(history),
              'Còn chờ':len(service.queue)-len(history),'Được phép training':0})
    pending = st.checkbox('Chỉ ảnh tôi chưa xử lý',value=True)
    identities = [k for k in service.queue if not pending or k not in history]
    if identities:
        identity = st.selectbox('Ảnh cần review',identities,format_func=lambda k:service.queue[k]['image_name'])
        row,image,initial = service.inspect(token,identity)
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
            service.submit(token,identity,current,decision,evidence,attest,negative)
            st.success('Đã lưu annotation version mới trong private storage; nhãn gốc giữ nguyên.')
            st.rerun()
    if st.button('Xuất bbox đã phê duyệt của tôi'):
        st.success(str(service.export(token)))
