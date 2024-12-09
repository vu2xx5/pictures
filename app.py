import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from edit_picture.edit import Edit
from utils import image_files
from PIL import Image

st.title("Ứng dụng chỉnh sửa ảnh và nhận diện khuôn mặt")
upload_image = st.file_uploader("Tải ảnh lên:", type=["jpg", "png", "jpeg"])  # tải ảnh lên

if upload_image is not None:
    cv2_image = image_files.open_image(upload_image)
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB cho Streamlit
    edit = Edit(cv2_image)
    st.image(image_rgb, caption="Ảnh gốc", use_column_width=True)

    option = st.selectbox("Chọn thao tác", ["Chỉnh sửa độ sáng", "Cắt ảnh", "Xoay ảnh", "Làm mờ"])

    if option == "Chỉnh sửa độ sáng":
        brightness = st.slider("Điều chỉnh độ sáng:", 0.5, 3.0, 1.0)
        image_brightness = edit.brightness_image(brightness)
        st.image(cv2.cvtColor(image_brightness, cv2.COLOR_BGR2RGB), caption="Ảnh chỉnh độ sáng", use_column_width=True)
        st.download_button(
                label="Tải ảnh",
                data=image_brightness.tobytes(),
                file_name="image_brightness.png",
                mime="image/png",
            )
    elif option == "Xoay ảnh":
        angle = st.slider("Chọn góc xoay (độ):", 0, 270, 0, step=90)
        rotated_image = edit.rotate_image(angle)
        st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption=f"Ảnh xoay {angle}°", use_column_width=True)
        st.download_button(
                label="Tải ảnh",
                data=rotated_image.tobytes(),
                file_name="rotated_image.png",
                mime="image/png",
            )
    elif option == "Làm mờ":
        ksize = st.slider("Chọn kích thước kernel:", 3, 11, 5, step=2)
        blurred_image = edit.blur_image(ksize)
        st.image(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB), caption="Ảnh làm mờ", use_column_width=True)
        st.download_button(
                label="Tải ảnh",
                data=blurred_image.tobytes(),
                file_name="blurred_image.png",
                mime="image/png",
            )
    else:
        st.write("Kéo để chọn vùng muốn cắt:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Màu nền
            stroke_width=2,                      # Độ dày viền
            stroke_color="#ff0000",              # Màu viền
            background_image=Image.open(upload_image),      
            update_streamlit=True,
            height=cv2_image.shape[0],           # Chiều cao ảnh
            width=cv2_image.shape[1],            # Chiều rộng ảnh
            drawing_mode="rect",                 # Chế độ vẽ hình chữ nhật
            key="canvas",
        )
    
        # Xử lý vùng được chọn
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                coords = objects[-1]["left"], objects[-1]["top"], objects[-1]["width"], objects[-1]["height"]
                x0, y0, w, h = map(int, coords)
                cropped_image = cv2_image[y0:y0 + h, x0:x0 + w]  # Cắt ảnh bằng slicing
                
                # Hiển thị ảnh sau khi cắt
                st.write("Ảnh sau khi cắt:")
                st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption="Ảnh đã cắt", use_column_width=True)
                st.download_button(
                label="Tải ảnh",
                data=cropped_image.tobytes(),
                file_name="cropped_image.png",
                mime="image/png",
                )